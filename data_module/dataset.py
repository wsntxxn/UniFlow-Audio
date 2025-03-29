from pathlib import Path
from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, Sequence
import json
import pickle

from tqdm import tqdm
import numpy as np
from h5py import File
import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision
import random

from utils.diffsinger_utilities import norm_interp_f0


def read_jsonl_to_mapping(
    jsonl_file: str | Path, key_col: str, value_col: str
) -> dict[str, str]:
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
    TODO handle duplicate keys
    """
    mapping = {}
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            key = data[key_col]
            value = data[value_col]
            mapping[key] = value
    return mapping


def read_from_h5(key: str, h5_path: str, cache: dict[str, str] | None = None):
    if cache is None:
        with File(h5_path, "r") as reader:
            return reader[key][()]
    else:
        if h5_path not in cache:
            cache[h5_path] = File(h5_path, "r")
        return cache[h5_path][key][()]
    
def read_from_h5_tmp(key: str, h5_path: str, cache: dict[str, str] | None = None):
    if cache is None:
        # with File(h5_path, "r") as reader:
        #     return reader
        return File(h5_path, "r")
    else:
        if h5_path not in cache:
            cache[h5_path] = File(h5_path, "r")
        return cache[h5_path]


@dataclass(kw_only=True)
class HDF5DatasetMixin:
    def __post_init__(self) -> None:
        self.h5_cache: dict[str, File] = {}

    def __del__(self) -> None:
        for h5_file in self.h5_cache.values():
            if h5_file:
                try:
                    h5_file.close()
                except:
                    pass


@dataclass(kw_only=True)
class AudioWaveformDataset(HDF5DatasetMixin):

    target_sr: int | None = None
    use_h5_cache: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.h5_src_sr_map = {}

    def load_waveform(self, audio_id: str, audio_path: str):
        if audio_path.endswith(".hdf5") or audio_path.endswith(".h5"):
            # on guizhou file system, using cached h5py.File will cause OOM error
            if self.use_h5_cache:
                waveform = read_from_h5(audio_id, audio_path, self.h5_cache)
            else:
                waveform = read_from_h5(audio_id, audio_path)
            if audio_path not in self.h5_src_sr_map:
                with File(audio_path, "r") as hf:
                    self.h5_src_sr_map[audio_path] = hf["sample_rate"][()]
            orig_sr = self.h5_src_sr_map[audio_path]
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        else:
            waveform, orig_sr = torchaudio.load(audio_path)
            # average multi-channel to single-channel
            waveform = waveform.mean(0)

        if self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=self.target_sr
            )
        return waveform


@dataclass
class AudioGenerationDataset(AudioWaveformDataset):

    content: str | Path
    audio: str | Path | None = None
    condition: str | Path | None = None
    id_col: str = "audio_id"
    id_col_in_content: str | None = None
    content_col: str = "content"
    id_col_in_audio: str | None = None
    audio_col: str = "audio"
    id_col_in_condition: str | None = None
    condition_col: str = "condition"

    # TODO how to add instructions of the condition, like `condition_name` or `task_name`
    # and then map `xx_name` to specific prompts?

    def __post_init__(self, ):
        super().__post_init__()

        id_col_in_content = self.id_col_in_content or self.id_col
        self.id_to_content = read_jsonl_to_mapping(
            self.content, id_col_in_content, self.content_col
        )
        # id_to_content: {'id1': '<caption1>', 'id2': '<caption2>'}

        id_col_in_audio = self.id_col_in_audio or self.id_col
        if self.audio:
            self.id_to_audio = read_jsonl_to_mapping(
                self.audio, id_col_in_audio, self.audio_col
            )
        else:
            self.id_to_audio = None
        # id_to_audio: {'id1': '<audio path1>', 'id2': '<audio path2>'}

        if self.condition:
            id_col_in_condition = self.id_col_in_condition or self.id_col
            self.id_to_condition = read_jsonl_to_mapping(
                self.condition, id_col_in_condition, self.condition_col
            )
        else:
            self.id_to_condition = None

        self.audio_ids = list(self.id_to_content.keys())

    @property
    @abstractmethod
    def task(self):
        ...

    def __len__(self) -> int:
        return len(self.audio_ids)

    @abstractmethod
    def load_condition(self, audio_id: str, condition_path: str) -> Any:
        ...

    @abstractmethod
    def load_content(self, audio_id: str, content_or_path: str) -> Any:
        ...

    @abstractmethod
    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        ...

    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        content = self.load_content(audio_id, content_or_path)

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            waveform = self.load_waveform(audio_id, audio_path)
        else:  # inference, only content is available
            waveform = None

        duration = self.load_duration(content, waveform)

        return content, waveform, duration

    def __getitem__(self, index) -> dict[str, Any]:
        audio_id = self.audio_ids[index]
        if self.task == "video_to_audio":
            content, waveform, duration, label = self.load_content_waveform(audio_id)
        else:
            content, waveform, duration = self.load_content_waveform(audio_id)
            label = None

        if self.id_to_condition:
            condition_path = self.id_to_condition[audio_id]
            condition = self.load_condition(audio_id, condition_path)
        else:
            condition = None

        return {
            "audio_id": audio_id,
            "content": content,
            "label": label,
            "waveform": waveform,
            "condition": condition,
            "duration": duration,
            "task": self.task
        }


@dataclass
class TextToAudioDataset(AudioGenerationDataset):

    content_col: str = "caption"

    @property
    def task(self):
        return "text_to_audio"

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        return [1.0]  # dummy duration sequence for batchify

    def load_content(self, audio_id: str, content_or_path: str):
        # text-to-audio / text-to-music, directly use text as the content input
        return content_or_path


@dataclass
class VideoToAudioDataset(AudioGenerationDataset):

    content_col: str = "video"
    total_duration: int = 10.0

    def __post_init__(self, ):
        super().__post_init__()

    def load_content_waveform(self, audio_id: str):
        video_path = self.id_to_content[audio_id]

        if video_path.endswith(".hdf5") or video_path.endswith(".h5"):
            # on guizhou file system, using cached h5py. File will cause OOM error
            if self.use_h5_cache:
                vggsound = read_from_h5_tmp(audio_id, video_path, self.h5_cache)
            else:
                vggsound = read_from_h5_tmp(audio_id, video_path)

            id_set = vggsound["Video_id"][:]
            idx = np.where(id_set == audio_id.encode())

            video_features = vggsound["Video"][idx]
            if video_features.ndim==3:
                # clip features
                video_features = torch.as_tensor(video_features, dtype=torch.float32).squeeze(0)
            elif video_features.ndim==4:
                # cavp features
                video_features = torch.as_tensor(video_features, dtype=torch.float32).squeeze(0).squeeze(0)
            else:
                raise ValueError("video features n_dim is not supported !")
            label = vggsound["Label"][idx]

            if self.id_to_audio:
                waveform = vggsound["Audio"][idx]
                waveform =  torch.as_tensor(waveform, dtype=torch.float32).squeeze(0)
            else:  # inference, only content is available
                waveform = None
        else:
            raise NotImplementedError("VGGSound must be load by h5 file !")        

        duration = self.load_duration(video_features, waveform)

        return video_features, waveform, duration, label

    def load_duration(self, content: Any, waveform: torch.Tensor) -> Sequence[float]:
        L = content.shape[0]
        duration = torch.full( (L,), self.total_duration) / L
        return duration

    @property
    def task(self):
        return "video_to_audio"


class TextToSpeechDataset(AudioWaveformDataset):
    ...


class SpeechEnhancementDataset(AudioWaveformDataset):
    ...


@dataclass
class OpenCpopSingingDataset(AudioGenerationDataset):

    content_col: str = "midi"

    @property
    def task(self):
        return "singing_voice_synthesis"

    def load_content(self, audio_id: str, content_or_path: str):
        with open(content_or_path, "rb") as file:
            midi = pickle.load(file)[audio_id]
        return midi

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        return content["phoneme_duration"].astype(np.float32)


@dataclass(kw_only=True)
class PopCsSingingDataset(AudioGenerationDataset):

    content_col: str = "phone_pitch"
    f0_stats: str
    pitch_norm: str = "log"
    use_uv: bool = True
    max_duration: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self.f0_mean, self.f0_std = np.load(self.f0_stats)
        self.f0_mean = float(self.f0_mean)
        self.f0_std = float(self.f0_std)

    @property
    def task(self):
        return "singing_acoustic_modeling"

    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        with File(content_or_path, "r") as hf:
            phoneme = hf["phoneme"][audio_id][()]
            phoneme_duration = hf["phoneme_duration"][audio_id][()].astype(
                np.float32
            )
            f0 = hf["f0"][audio_id][()].astype(np.float32)

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            waveform = self.load_waveform(audio_id, audio_path)
        else:  # inference, only content is available
            waveform = None

        f0, uv = norm_interp_f0(
            f0, self.f0_mean, self.f0_std, self.pitch_norm, self.use_uv
        )
        if self.max_duration is not None:
            cumsum_phone_duration = np.cumsum(phoneme_duration)
            overlength_idxs = np.where(
                cumsum_phone_duration >= self.max_duration
            )[0]
            if len(overlength_idxs) > 0:
                trunc_idx = overlength_idxs[0]
                phoneme = phoneme[:trunc_idx]
                phoneme_duration = phoneme_duration[:trunc_idx]
                trunc_duration = cumsum_phone_duration[trunc_idx - 1]
                orig_duration = cumsum_phone_duration[-1]
                f0 = f0[:int(trunc_duration / orig_duration * f0.shape[0])]
                uv = uv[:int(trunc_duration / orig_duration * uv.shape[0])]
                if waveform is not None:
                    waveform = waveform[:int(
                        trunc_duration / orig_duration * waveform.shape[0]
                    )]

        content = {
            "phoneme": phoneme,
            "phoneme_duration": phoneme_duration,
            "f0": f0,
            "uv": uv
        }
        duration = self.load_duration(content, waveform)

        return content, waveform, duration

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        return content["phoneme_duration"]


class AudioSuperResolutionDataset(AudioWaveformDataset):
       

    # lowpass audio in content.jsonl，audio_id+caption
    content: str | Path
    audio: str | Path | None = None
    downsampling_ratio: int | None
    condition: str | Path | None = None
    id_col: str = "audio_id"
    id_col_in_content: str | None = None
    content_col: str = "caption"
    id_col_in_audio: str | None = None
    audio_col: str = "audio"
    id_col_in_condition: str | None = None
    condition_col: str = "condition"

    def __post_init__(self, ):
        super().__post_init__()

        id_col_in_content = self.id_col_in_content or self.id_col
        self.id_to_content = read_jsonl_to_mapping(
            self.content, id_col_in_content, self.content_col
        )
        # id_to_content: {'id1': '<caption1>', 'id2': '<caption2>'}

        id_col_in_audio = self.id_col_in_audio or self.id_col
        if self.audio:
            self.id_to_audio = read_jsonl_to_mapping(
                self.audio, id_col_in_audio, self.audio_col
            )
        else:
            self.id_to_audio = None
        # id_to_audio: {'id1': '<audio path1>', 'id2': '<audio path2>'}

        
        condition = None

        self.audio_ids = list(self.id_to_content.keys())


    @property
    @abstractmethod
    def task(self):
        return "audio_super_resolution"

    def __len__(self) -> int:
        return len(self.audio_ids)

    
    def load_content(self, audio_id: str, content_or_path: str) -> Any:
        waveform, sr = torchaudio.load(content_or_path)
        return waveform
    
    def load_duration(self, content: Any, waveform: torch.Tensor) -> Sequence[float]:
        if content.dim() == 1:
            duration_time = content.size(0)//self.downsampling_ratio
        else:
            duration_time = content.size(1)//self.downsampling_ratio
        duration_value =   self.downsampling_ratio / self.target_sr 
        duration = np.full(duration_time, duration_value)
        return duration

    
    
    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        content = self.load_content(audio_id, content_or_path)
        content=content.mean(0)
        max_length = 250 * 480

        # clip can be omitted
        if len(content) > max_length:
            start_index = random.randint(0, len(content) - max_length)
            content = content[start_index:start_index + max_length]


        if self.id_to_audio:  # training, audio is the target
            #audio_path = self.base_path +'/' + self.id_to_audio[audio_id]
            audio_path = self.id_to_audio[audio_id]
            waveform = self.load_waveform(audio_id, audio_path)
            if len(waveform) > max_length:
                waveform = waveform[start_index:start_index+max_length]
        else:  # inference, only content is available
            waveform = None
        
        
        duration = self.load_duration(content, waveform)
        
        

        return content, waveform, duration
    def __getitem__(self, index) -> dict[str, Any]:
        
        audio_id = self.audio_ids[index]


        low_res_waveform, high_res_waveform, duration= self.load_content_waveform(audio_id)
        condition= None
        content_length = torch.tensor(len(low_res_waveform))
        content_dict = {"content":low_res_waveform, "content_length":content_length}
        
        return {
            "audio_id": audio_id,
            "content": content_dict,
            "waveform": high_res_waveform,
            "condition": condition,
            "duration": duration,
            "task": self.task,
        }
    
    def __del__(self):
        if hasattr(self, 'h5_cache'):
            for h5_file in self.h5_cache.values():
                h5_file.close()


class AudioGenConcatDataset(Dataset):
    def __init__(self, datasets: list[AudioGenerationDataset]):
        self.datasets = datasets
        self.lengths = np.array([len(d) for d in datasets])
        self.cum_sum_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cum_sum_lengths - 1, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sum_lengths[dataset_idx - 1]
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx]


if __name__ == '__main__':

    from tqdm import tqdm

    dataset = AudioGenConcatDataset(
        datasets=[
            # TextToAudioDataset(
            #     content="./data/audiocaps/test/caption.jsonl",
            #     audio="./data/audiocaps/test/audio.jsonl",
            #     target_sr=24000
            # ),
            PopCsSingingDataset(
                content="./data/popcs/train/phone_pitch.jsonl",
                audio="./data/popcs/train/audio.jsonl",
                target_sr=24000,
                f0_stats="./data/popcs/train/f0_mean_std.npy",
                pitch_norm="log",
                use_uv=True,
                max_duration=20.0
            )
        ]
    )

    from data_module.collate_function import PaddingCollate

    collate_fn = PaddingCollate(pad_keys=["waveform", "duration", "f0", "uv"])
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate_fn, batch_size=4
    )

    for item in tqdm(dataset):
        # for batch in tqdm(dataloader):
        duration = item["duration"]
        pass
