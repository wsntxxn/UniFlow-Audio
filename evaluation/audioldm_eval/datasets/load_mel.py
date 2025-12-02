import torch
import os
import numpy as np
import torchaudio
from tqdm import tqdm
# import librosa


def pad_short_audio(audio, min_samples=32000):
    if (audio.size(-1) < min_samples):
        audio = torch.nn.functional.pad(
            audio, (0, min_samples - audio.size(-1)),
            mode='constant',
            value=0.0
        )
    return audio


class MelPairedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data1: str | dict,
        data2: str | dict,
        _stft,
        sr=16000,
        fbin_mean=None,
        fbin_std=None,
        augment=False,
        limit_num=None,
    ):
        if isinstance(data1, str):
            self.aid_to_audio1 = {
                x: os.path.join(data1, x)
                for x in os.listdir(data1)
            }
        else:
            self.aid_to_audio1 = data1
        self.aids1 = sorted(self.aid_to_audio1.keys())

        if isinstance(data2, str):
            self.aid_to_audio2 = {
                x: os.path.join(data1, x)
                for x in os.listdir(data1)
            }
        else:
            self.aid_to_audio2 = data2
        self.aids2 = sorted(self.aid_to_audio2.keys())

        if limit_num is not None:
            self.aids1 = self.aids1[:limit_num]
            self.aids2 = self.aids2[:limit_num]

        self.align_two_file_list()

        self._stft = _stft
        self.sr = sr
        self.augment = augment

        # if fbin_mean is not None:
        #     self.fbin_mean = fbin_mean[..., None]
        #     self.fbin_std = fbin_std[..., None]
        # else:
        #     self.fbin_mean = None
        #     self.fbin_std = None

    def align_two_file_list(self):

        keyset1 = set(self.aids1)
        keyset2 = set(self.aids2)

        intersect_keys = keyset1.intersection(keyset2)

        self.aids = list(intersect_keys)

        print(f"Two path have {len(intersect_keys)} intersection files")

    def __getitem__(self, index):
        while True:
            try:
                aid = self.aids[index]
                filename1 = self.aid_to_audio1[aid]
                filename2 = self.aid_to_audio2[aid]
                mel1, _, audio1 = self.get_mel_from_file(filename1)
                mel2, _, audio2 = self.get_mel_from_file(filename2)
                break
            except Exception as e:
                print(index, e)
                index = (index + 1) % len(self.aids)

        # if(self.fbin_mean is not None):
        #     mel = (mel - self.fbin_mean) / self.fbin_std
        min_len = min(mel1.shape[-1], mel2.shape[-1])
        return (
            mel1[..., :min_len],
            mel2[..., :min_len],
            aid,
            (audio1, audio2),
        )

    def __len__(self):
        return len(self.aids)

    def get_mel_from_file(self, audio_file):
        audio, file_sr = torchaudio.load(audio_file)
        # Only use the first channel
        audio = audio[0:1, ...]
        audio = audio - audio.mean()
        if file_sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=file_sr, new_freq=self.sr
            )

        if self._stft is not None:
            melspec, energy = self.get_mel_from_wav(audio[0, ...])
        else:
            melspec, energy = None, None

        return melspec, energy, audio

    def get_mel_from_wav(self, audio):
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)

        # =========================================================================
        # Following the processing in https://github.com/v-iashin/SpecVQGAN/blob/5bc54f30eb89f82d129aa36ae3f1e90b60e73952/vocoder/mel2wav/extract_mel_spectrogram.py#L141
        melspec, energy = self._stft.mel_spectrogram(
            audio, normalize_fun=torch.log10
        )
        melspec = (melspec * 20) - 20
        melspec = (melspec + 100) / 100
        melspec = torch.clip(melspec, min=0, max=1.0)
        # =========================================================================
        # Augment
        # if(self.augment):
        #     for i in range(1):
        #         random_start = int(torch.rand(1) * 950)
        #         melspec[0,:,random_start:random_start+50] = 0.0
        # =========================================================================
        melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        return melspec, energy


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: str | dict,
        sr: int = 16000,
        limit_num: int | None = None,
    ):
        if isinstance(data, str):
            self.aid_to_audio = {
                x: os.path.join(data, x)
                for x in os.listdir(data)
            }
        elif isinstance(data, dict):
            self.aid_to_audio = data
        self.aids = sorted(list(self.aid_to_audio.keys()))
        if limit_num is not None:
            self.aids = self.aids[:limit_num]
        self.sr = sr

    def __getitem__(self, index):
        while True:
            try:
                aid = self.aids[index]
                filename = self.aid_to_audio[aid]
                waveform = self.read_from_file(filename)
                if waveform.size(-1) < 1:
                    raise ValueError("empty file %s" % filename)
                break
            except Exception as e:
                print(index, e)
                index = (index + 1) % len(self.datalist)

        return waveform, aid

    def __len__(self):
        return len(self.aids)

    def read_from_file(self, audio_file):
        audio, file_sr = torchaudio.load(audio_file)
        # Only use the first channel
        audio = audio[0:1, ...]
        audio = audio - audio.mean()

        # if file_sr != self.sr and file_sr == 32000 and self.sr == 16000:
        #     audio = audio[..., ::2]
        # if file_sr != self.sr and file_sr == 48000 and self.sr == 16000:
        #     audio = audio[..., ::3]
        # el

        if file_sr != self.sr:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=file_sr,
                new_freq=self.sr,  # rolloff=0.95, lowpass_filter_width=16 
            )
            # audio = torch.FloatTensor(librosa.resample(audio.numpy(), file_sr, self.sr))

        audio = pad_short_audio(audio, min_samples=32000)
        return audio


def load_npy_data(loader):
    new_train = []
    for mel, waveform, filename in tqdm(loader):
        batch = batch.float().numpy()
        new_train.append(batch.reshape(-1, ))
    new_train = np.array(new_train)
    return new_train


if __name__ == "__main__":
    path = "/scratch/combined/result/ground/00294 harvest festival rumour 1_mel.npy"
    temp = np.load(path)
    print("temp", temp.shape)
