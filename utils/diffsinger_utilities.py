import six
from pathlib import Path
import re
import json
from collections import OrderedDict
from typing import Union

import numpy as np
import librosa
import torch

PAD = "<pad>"
EOS = "<EOS>"
UNK = "<UNK>"
SEG = "|"
RESERVED_TOKENS = [PAD, EOS, UNK]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
UNK_ID = RESERVED_TOKENS.index(UNK)  # Normally 2

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 /
                     700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0
          ] = (f0_mel[f0_mel > 0] -
               F0_MEL_MIN) * (F0_BIN - 2) / (F0_MEL_MAX - F0_MEL_MIN) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > F0_BIN - 1] = F0_BIN - 1
    f0_coarse = (f0_mel +
                 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(), f0_coarse.min()
    )
    return f0_coarse


def norm_f0(
    f0: Union[np.ndarray, torch.Tensor],
    uv: Union[None, np.ndarray],
    f0_mean: float,
    f0_std: float,
    pitch_norm: str = "log",
    use_uv: bool = True
):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = (f0 - f0_mean) / f0_std
    if pitch_norm == 'log':
        f0 = torch.log2(f0) if is_torch else np.log2(f0)
    if uv is not None and use_uv:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(
    f0: Union[np.ndarray, torch.Tensor],
    f0_mean: float,
    f0_std: float,
    pitch_norm: str = "log",
    use_uv: bool = True
):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, f0_mean, f0_std, pitch_norm, use_uv)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    uv = torch.as_tensor(uv).float()
    f0 = torch.as_tensor(f0).float()
    if is_torch:
        f0 = f0.to(device)
    return f0, uv


def denorm_f0(
    f0,
    uv,
    pitch_norm="log",
    f0_mean=None,
    f0_std=None,
    pitch_padding=None,
    min=None,
    max=None,
    use_uv=True
):
    if pitch_norm == 'standard':
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == 'log':
        f0 = 2**f0
    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and use_uv:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def librosa_pad_lr(x, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


def get_pitch(
    wav_file: Union[str, Path], sample_rate: int, frame_shift: float
):
    import parselmouth
    hop_size = int(frame_shift * sample_rate)
    wav, _ = librosa.core.load(wav_file, sr=sample_rate)
    # l_pad, r_pad = librosa_pad_lr(wav, hop_size, 1)
    # wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)

    latent_length = wav.shape[0] // hop_size
    f0_min = 80
    f0_max = 750
    pad_size = 4

    f0 = parselmouth.Sound(wav, sample_rate).to_pitch_ac(
        time_step=frame_shift,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max
    ).selected_array['frequency']
    delta_l = latent_length - len(f0)
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse


def remove_empty_lines(text):
    """remove empty lines"""
    assert (len(text) > 0)
    assert (isinstance(text, list))
    text = [t.strip() for t in text]
    if "" in text:
        text.remove("")
    return text


def is_sil_phoneme(p):
    return not p[0].isalpha()


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""
    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        """Transform a human-readable string into a sequence of int ids.

        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
        s: human-readable string to be converted.

        Returns:
        ids: list of integers
        """
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string.

        EOS is not expected in ids.

        Args:
        ids: list of integers to be converted.
        strip_extraneous: bool, whether to strip off extraneous tokens
            (EOS and PAD).

        Returns:
        s: human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions.

        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
        ids: list of integers to be converted.

        Returns:
        strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


class TokenTextEncoder(TextEncoder):
    """Encoder based on a user-supplied vocabulary (file or list)."""
    def __init__(
        self,
        vocab_filename,
        reverse=False,
        vocab_list=None,
        replace_oov=None,
        num_reserved_ids=NUM_RESERVED_TOKENS
    ):
        """Initialize from a file or list, one token per line.

        Handling of reserved tokens works as follows:
        - When initializing from a list, we add reserved tokens to the vocab.
        - When initializing from a file, we do not add reserved tokens to the vocab.
        - When saving vocab files, we save reserved tokens to the file.

        Args:
            vocab_filename: If not None, the full filename to read vocab from. If this
                is not None, then vocab_list should be None.
            reverse: Boolean indicating if tokens should be reversed during encoding
                and decoding.
            vocab_list: If not None, a list of elements of the vocabulary. If this is
                not None, then vocab_filename should be None.
            replace_oov: If not None, every out-of-vocabulary token seen when
                encoding will be replaced by this string (which must be in vocab).
            num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
        """
        super(TokenTextEncoder,
              self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
            self._init_vocab_from_file(vocab_filename)
        else:
            assert vocab_list is not None
            self._init_vocab_from_list(vocab_list)
        self.pad_index = self._token_to_id[PAD]
        self.eos_index = self._token_to_id[EOS]
        self.unk_index = self._token_to_id[UNK]
        self.seg_index = self._token_to_id[
            SEG] if SEG in self._token_to_id else self.eos_index

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [
                t if t in self._token_to_id else self._replace_oov
                for t in tokens
            ]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids, strip_eos=False, strip_padding=False):
        if strip_padding and self.pad() in list(ids):
            pad_pos = list(ids).index(self.pad())
            ids = ids[:pad_pos]
        if strip_eos and self.eos() in list(ids):
            eos_pos = list(ids).index(self.eos())
            ids = ids[:eos_pos]
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def __len__(self):
        return self.vocab_size

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    def _init_vocab_from_file(self, filename):
        """Load vocab from a file.

        Args:
        filename: The file to load vocabulary from.
        """
        with open(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token

        self._init_vocab(token_gen(), add_reserved_tokens=False)

    def _init_vocab_from_list(self, vocab_list):
        """Initialize tokens from a list of tokens.

        It is ok if reserved tokens appear in the vocab list. They will be
        removed. The set of tokens in vocab_list should be unique.

        Args:
        vocab_list: A list of tokens.
        """
        def token_gen():
            for token in vocab_list:
                if token not in RESERVED_TOKENS:
                    yield token

        self._init_vocab(token_gen())

    def _init_vocab(self, token_generator, add_reserved_tokens=True):
        """Initialize vocabulary with tokens from token_generator."""

        self._id_to_token = {}
        non_reserved_start_index = 0

        if add_reserved_tokens:
            self._id_to_token.update(enumerate(RESERVED_TOKENS))
            non_reserved_start_index = len(RESERVED_TOKENS)

        self._id_to_token.update(
            enumerate(token_generator, start=non_reserved_start_index)
        )

        # _token_to_id is the reverse of _id_to_token
        self._token_to_id = dict(
            (v, k) for k, v in six.iteritems(self._id_to_token)
        )

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def seg(self):
        return self.seg_index

    def store_to_file(self, filename):
        """Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
        filename: Full path of the file to store the vocab to.
        """
        with open(filename, "w") as f:
            for i in range(len(self._id_to_token)):
                f.write(self._id_to_token[i] + "\n")

    def sil_phonemes(self):
        return [p for p in self._id_to_token.values() if not p[0].isalpha()]


class TextGrid(object):
    def __init__(self, text):
        text = remove_empty_lines(text)
        self.text = text
        self.line_count = 0
        self._get_type()
        self._get_time_intval()
        self._get_size()
        self.tier_list = []
        self._get_item_list()

    def _extract_pattern(self, pattern, inc):
        """
        Parameters
        ----------
        pattern : regex to extract pattern
        inc : increment of line count after extraction
        Returns
        -------
        group : extracted info
        """
        try:
            group = re.match(pattern, self.text[self.line_count]).group(1)
            self.line_count += inc
        except AttributeError:
            raise ValueError(
                "File format error at line %d:%s" %
                (self.line_count, self.text[self.line_count])
            )
        return group

    def _get_type(self):
        self.file_type = self._extract_pattern(r"File type = \"(.*)\"", 2)

    def _get_time_intval(self):
        self.xmin = self._extract_pattern(r"xmin = (.*)", 1)
        self.xmax = self._extract_pattern(r"xmax = (.*)", 2)

    def _get_size(self):
        self.size = int(self._extract_pattern(r"size = (.*)", 2))

    def _get_item_list(self):
        """Only supports IntervalTier currently"""
        for itemIdx in range(1, self.size + 1):
            tier = OrderedDict()
            item_list = []
            tier_idx = self._extract_pattern(r"item \[(.*)\]:", 1)
            tier_class = self._extract_pattern(r"class = \"(.*)\"", 1)
            if tier_class != "IntervalTier":
                raise NotImplementedError(
                    "Only IntervalTier class is supported currently"
                )
            tier_name = self._extract_pattern(r"name = \"(.*)\"", 1)
            tier_xmin = self._extract_pattern(r"xmin = (.*)", 1)
            tier_xmax = self._extract_pattern(r"xmax = (.*)", 1)
            tier_size = self._extract_pattern(r"intervals: size = (.*)", 1)
            for i in range(int(tier_size)):
                item = OrderedDict()
                item["idx"] = self._extract_pattern(r"intervals \[(.*)\]", 1)
                item["xmin"] = self._extract_pattern(r"xmin = (.*)", 1)
                item["xmax"] = self._extract_pattern(r"xmax = (.*)", 1)
                item["text"] = self._extract_pattern(r"text = \"(.*)\"", 1)
                item_list.append(item)
            tier["idx"] = tier_idx
            tier["class"] = tier_class
            tier["name"] = tier_name
            tier["xmin"] = tier_xmin
            tier["xmax"] = tier_xmax
            tier["size"] = tier_size
            tier["items"] = item_list
            self.tier_list.append(tier)

    def toJson(self):
        _json = OrderedDict()
        _json["file_type"] = self.file_type
        _json["xmin"] = self.xmin
        _json["xmax"] = self.xmax
        _json["size"] = self.size
        _json["tiers"] = self.tier_list
        return json.dumps(_json, ensure_ascii=False, indent=2)


def read_duration_from_textgrid(
    textgrid_path: Union[str, Path],
    phoneme: str,
    utterance_duration: float,
):
    ph_list = phoneme.split(" ")
    with open(textgrid_path, "r") as f:
        textgrid = f.readlines()
    textgrid = remove_empty_lines(textgrid)
    textgrid = TextGrid(textgrid)
    textgrid = json.loads(textgrid.toJson())

    split = np.ones(len(ph_list) + 1, np.float32) * -1
    tg_idx = 0
    ph_idx = 0
    tg_align = [x for x in textgrid['tiers'][-1]['items']]
    tg_align_ = []
    for x in tg_align:
        x['xmin'] = float(x['xmin'])
        x['xmax'] = float(x['xmax'])
        if x['text'] in ['sil', 'sp', '', 'SIL', 'PUNC', '<SP>', '<AP>']:
            x['text'] = ''
            if len(tg_align_) > 0 and tg_align_[-1]['text'] == '':
                tg_align_[-1]['xmax'] = x['xmax']
                continue
        tg_align_.append(x)
    tg_align = tg_align_
    tg_len = len([x for x in tg_align if x['text'] != ''])
    ph_len = len([x for x in ph_list if not is_sil_phoneme(x)])
    assert tg_len == ph_len, (tg_len, ph_len, tg_align, ph_list, textgrid_path)
    while tg_idx < len(tg_align) or ph_idx < len(ph_list):
        if tg_idx == len(tg_align) and is_sil_phoneme(ph_list[ph_idx]):
            split[ph_idx] = 1e8
            ph_idx += 1
            continue
        x = tg_align[tg_idx]
        if x['text'] == '' and ph_idx == len(ph_list):
            tg_idx += 1
            continue
        assert ph_idx < len(ph_list), (
            tg_len, ph_len, tg_align, ph_list, textgrid_path
        )

        ph = ph_list[ph_idx]
        if x['text'] == '' and not is_sil_phoneme(ph):
            assert False, (ph_list, tg_align)
        if x['text'] != '' and is_sil_phoneme(ph):
            ph_idx += 1
        else:
            assert (x['text'] == '' and is_sil_phoneme(ph)) \
                   or x['text'].lower() == ph.lower() \
                   or x['text'].lower() == 'sil', (x['text'], ph)
            split[ph_idx] = x['xmin']
            if ph_idx > 0 and split[ph_idx - 1] == -1 and is_sil_phoneme(
                ph_list[ph_idx - 1]
            ):
                split[ph_idx - 1] = split[ph_idx]
            ph_idx += 1
            tg_idx += 1
    assert tg_idx == len(tg_align), (tg_idx, [x['text'] for x in tg_align])
    assert ph_idx >= len(ph_list) - 1, (
        ph_idx, ph_list, len(ph_list), [x['text']
                                        for x in tg_align], textgrid_path
    )

    split[0] = 0
    split[-1] = utterance_duration
    duration = np.diff(split)
    return duration
