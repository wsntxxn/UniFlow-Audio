from pathlib import Path
from abc import abstractmethod
from typing import Any
import json
import numpy as np
from h5py import File
import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision


def read_jsonl_to_mapping(
    jsonl_file: str | Path, key_col: str, value_col: str
) -> dict[str, str]:
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
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


class HDF5DatasetMixin:
    def __init__(self):
        self.h5_cache: dict[str, File] = {}

    def __del__(self):
        for h5_file in self.h5_cache.values():
            if h5_file:
                try:
                    h5_file.close()
                except:
                    pass


class AudioWaveformDataset(HDF5DatasetMixin):
    def __init__(
        self, target_sr: int | None = None, use_h5_cache: bool = True
    ):
        super().__init__()
        self.target_sr = target_sr
        self.h5_src_sr_map = {}
        self.use_h5_cache = use_h5_cache

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


class AudioGenerationDataset(AudioWaveformDataset):
    def __init__(
        self,
        content: str | Path,
        audio: str | Path,
        condition: str | Path | None = None,
        id_col: str = "audio_id",
        id_col_in_content: str | None = None,
        content_col: str = "content",
        id_col_in_audio: str | None = None,
        audio_col: str = "audio",
        id_col_in_condition: str | None = None,
        condition_col: str = "condition",
        # TODO how to add instructions of the condition, like `condition_name` or `task_name`
        # and then map `xx_name` to specific prompts?
        sample_rate: int | None = None,
        use_h5_cache: bool = True
    ):
        AudioWaveformDataset.__init__(self, sample_rate, use_h5_cache)
        id_col_in_content = id_col_in_content or id_col
        self.id_to_content = read_jsonl_to_mapping(
            content, id_col_in_content, content_col
        )
        # id_to_content: {'id1': '<caption1>', 'id2': '<caption2>'}

        id_col_in_audio = id_col_in_audio or id_col
        self.id_to_audio = read_jsonl_to_mapping(
            audio, id_col_in_audio, audio_col
        )
        # id_to_audio: {'id1': '<audio path1>', 'id2': '<audio path2>'}

        if condition:
            id_col_in_condition = id_col_in_condition or id_col
            self.id_to_condition = read_jsonl_to_mapping(
                condition, id_col_in_condition, condition_col
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
    def load_content(self, content_or_path: str) -> Any:
        ...

    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        content = self.load_content(content_or_path)

        audio_path = self.id_to_audio[audio_id]
        waveform = self.load_waveform(audio_id, audio_path)

        return content, waveform

    def __getitem__(self, index) -> dict[str, Any]:
        audio_id = self.audio_ids[index]
        content, waveform = self.load_content_waveform(audio_id)

        if self.id_to_condition:
            condition_path = self.id_to_condition[audio_id]
            condition = self.load_condition(audio_id, condition_path)
        else:
            condition = None

        return {
            "content": content,
            "waveform": waveform,
            "condition": condition,
            "task": self.task
        }


class TextToAudioDataset(AudioGenerationDataset):
    def __init__(
        self,
        caption: str | Path,
        audio: str | Path,
        condition: str | Path | None = None,
        sample_rate: int | None = None,
        use_h5_cache: bool = True
    ):
        super().__init__(
            caption,
            audio,
            condition,
            content_col="caption",
            sample_rate=sample_rate,
            use_h5_cache=use_h5_cache
        )

    @property
    def task(self):
        return "text_to_audio"

    def load_content(self, content_or_path: str):
        # text-to-audio / text-to-music, directly use text as the content input
        return content_or_path


class VideoToAudioDataset(AudioGenerationDataset):
    def __init__(
        self,
        video: str | Path,
        audio: str | Path,
        condition: str | Path | None = None,
        video_fps: int | None = None,
        video_size: tuple[int, int] = (256, 256),
        audio_sample_rate: int | None = None,
        use_h5_cache: bool = True
    ):
        super().__init__(
            video,
            audio,
            condition,
            content_col="video",
            sample_rate=audio_sample_rate,
            use_h5_cache=use_h5_cache
        )
        self.target_fps = video_fps
        self.resize_transform = torchvision.transforms.Resize(video_size)

    def load_content_waveform(self, audio_id: str):
        video_path = self.id_to_content[audio_id]
        video, waveform, meta = torchvision.io.read_video(video_path)
        # video: T x H x W x C, waveform: C x T
        orig_sr, fps = meta.get('audio_fps'), meta.get('video_fps')

        # average multi-channel to single-channel
        waveform = waveform.mean(0)
        # resample audio
        if self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=self.target_sr
            )

        # resample video
        if self.target_fps:
            video_resample_ratio = self.target_fps / fps
            new_length = int(round(video.shape[0] * video_resample_ratio))
            indices = torch.linspace(0, video.shape[0] - 1,
                                     steps=new_length).long()
            video = video[indices]

        # resize video
        video = self.resize_transform(
            video.permute(0, 3, 1, 2)
        )  # T x C x H x W

        return video, waveform

    @property
    def task(self):
        return "video_to_audio"


class TextToSpeechDataset(AudioWaveformDataset):
    ...


class SpeechEnhancementDataset(AudioWaveformDataset):
    ...


class SingingSynthesisDataset(AudioWaveformDataset):
    ...


class AudioSuperResolutionDataset(AudioWaveformDataset):
    ...


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
            TextToAudioDataset(
                caption="./clotho_val_caption.jsonl",
                audio="./clotho_val_audio.jsonl",
                sample_rate=16000
            ),
            TextToAudioDataset(
                caption="./audiocaps_test_caption.jsonl",
                audio="./audiocaps_test_audio.jsonl",
                sample_rate=16000
            ),
            VideoToAudioDataset(
                video="./vggsound_toy.jsonl",
                audio="./vggsound_toy.jsonl",
            )
        ]
    )
    # dataset = VideoToAudioDataset(
    #     video="./vggsound_toy.jsonl",
    #     audio="./vggsound_toy.jsonl",
    # )

    for item in tqdm(dataset):
        pass
