import json
import os
import pickle
import random
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torchaudio
from accel_hydra.utils.general import read_jsonl_to_mapping
from h5py import File
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import PHONEME2ID
from utils.diffsinger_utilities import TokenTextEncoder
from utils.petrel_oss import is_petrel_client_available, load_audio_from_petrel_oss

if is_petrel_client_available():
    from petrel_client.client import Client


def read_from_h5(
    key: str | None, h5_path: str, cache: dict[str, str] | None = None
):
    if cache is None:
        if key is None:
            return File(h5_path, "r")
        else:
            with File(h5_path, "r") as reader:
                return reader[key][()]
    else:
        if h5_path not in cache:
            cache[h5_path] = File(h5_path, "r")
        if key is None:
            return cache[h5_path]
        return cache[h5_path][key][()]


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
class TaskMixin:

    task_instruction: str | Path
    instruction_idx: int | None = None

    @property
    @abstractmethod
    def task(self):
        ...

    def __post_init__(self) -> None:
        self.task_to_num_instruction = {}
        with File(self.task_instruction, "r") as hf:
            for key in hf.keys():
                task, instruction_idx = key.rsplit("_", maxsplit=1)
                instruction_idx = int(instruction_idx)
                if task not in self.task_to_num_instruction:
                    self.task_to_num_instruction[task] = instruction_idx + 1
                else:
                    self.task_to_num_instruction[task] = max(
                        self.task_to_num_instruction[task], instruction_idx + 1
                    )
        # self.time_aligned = [False, False]
        # if self.task in self.time_aligned_tasks:
        #     self.time_aligned[0] = True
        # if self.task in self.non_time_aligned_tasks:
        #     self.time_aligned[1] = True
        # if self.task not in self.time_aligned_tasks + self.non_time_aligned_tasks:
        #     raise Exception(
        #         f"Time align property of {self.task} is not defined!"
        #     )


@dataclass(kw_only=True)
class AudioWaveformDataset(HDF5DatasetMixin):

    target_sr: int | None = None
    use_h5_cache: bool = True
    petrel_oss_config: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.h5_src_sr_map = {}
        if is_petrel_client_available() and self.petrel_oss_config:
            self.petrel_client = Client(self.petrel_oss_config)
        else:
            self.petrel_client = None

    def load_waveform(
        self, audio_id: str, audio_path: str, target_sr: int | None = None
    ):
        if audio_path.endswith(".hdf5") or audio_path.endswith(".h5"):
            try:
                # on guizhou file system, using cached h5py.File will cause OOM error
                if self.use_h5_cache:
                    waveform = read_from_h5(
                        audio_id, audio_path, self.h5_cache
                    )
                else:
                    waveform = read_from_h5(audio_id, audio_path)
                if audio_path not in self.h5_src_sr_map:
                    with File(audio_path, "r") as hf:
                        self.h5_src_sr_map[audio_path] = hf["sample_rate"][()]
                orig_sr = self.h5_src_sr_map[audio_path]
                waveform = torch.as_tensor(waveform, dtype=torch.float32)
            except Exception:
                print(f"Failed to load audio from {audio_path}")
                with open('./broken_audio_list.txt', 'a') as f:
                    f.write(audio_id + ',' + audio_path + '\n')
                return torch.zeros([100], dtype=torch.float32)
        elif audio_path.startswith("s3://"):  # from petrel OSS bucket
            waveform, orig_sr = load_audio_from_petrel_oss(
                audio_path, self.petrel_client
            )
            waveform = waveform.mean(0)
        else:  # from raw audio file
            try:
                waveform, orig_sr = torchaudio.load(audio_path)

            except Exception:
                print(f"Failed to load audio from {audio_path}")
                with open('./broken_audio_list.txt', 'a') as f:
                    f.write(audio_id + ',' + audio_path + '\n')
                return torch.zeros([100], dtype=torch.float32)
            # average multi-channel to single-channel
            waveform = waveform.mean(0)

        if target_sr:
            target_sr_ = target_sr
        else:
            target_sr_ = self.target_sr

        if target_sr_:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=target_sr_
            )
        return waveform


@dataclass
class AudioGenerationDataset(AudioWaveformDataset, TaskMixin):

    content: str | list[str]
    audio: str | list[str] | None = None
    condition: str | list[str] | None = None

    base_content_path: str | Path | None = None
    base_audio_path: str | Path | None = None
    base_condition_path: str | Path | None = None

    id_col: str = "audio_id"
    id_col_in_content: str | None = None
    content_col: str = "content"
    id_col_in_audio: str | None = None
    audio_col: str = "audio"
    id_col_in_condition: str | None = None
    condition_col: str = "condition"
    max_samples: int | None = None

    # TODO how to add instructions of the condition, like `condition_name` or `task_name`
    # and then map `xx_name` to specific prompts?

    def __post_init__(self, ):
        AudioWaveformDataset.__post_init__(self)
        TaskMixin.__post_init__(self)

        id_col_in_content = self.id_col_in_content or self.id_col
        self.id_to_content = read_jsonl_to_mapping(
            self.content, id_col_in_content, self.content_col
        )
        # id_to_content: {'id1': '<content1>', 'id2': '<content2>'}

        id_col_in_audio = self.id_col_in_audio or self.id_col
        if self.audio:
            self.id_to_audio = read_jsonl_to_mapping(
                self.audio, id_col_in_audio, self.audio_col
            )
            self.id_to_duration = read_jsonl_to_mapping(
                self.audio, id_col_in_audio, "duration"
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
        self.base_condition_path = Path(
            self.base_condition_path
        ) if self.base_condition_path else None

        self.audio_ids = list(self.id_to_content.keys())

        if self.max_samples is not None:
            # When the max_samples parameter is set, shuffling is enabled by default.
            random.shuffle(self.audio_ids)
            self.audio_ids = self.audio_ids[:min(
                len(self.audio_ids), self.max_samples
            )]

    def get_length(self, index: int) -> float:
        return self.id_to_duration[self.audio_ids[index]]

    def __len__(self) -> int:
        return len(self.audio_ids)

    @abstractmethod
    def load_condition(self, audio_id: str, condition_path: str) -> Any:
        ...

    @abstractmethod
    def load_content(self, audio_id: str,
                     content_or_path: str) -> tuple[Any, str]:
        ...

    @abstractmethod
    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        ...

    def load_content_waveform(
        self, audio_id: str
    ) -> tuple[Any, torch.Tensor, Sequence[float], str]:
        """
        Load content and waveform for the given audio_id.

        Args:
            audio_id: the unique id of the audio sample
        
        Returns:
            content: the content of the audio sample, can be any type, 
                normally a dict
            waveform: the waveform of the audio sample, None during inference
            duration: the duration sequence of the content for time-aligned 
                generation task; for non time-aligned task, return a dummy
                one [1.0]
            item_name: the interpretable name used in writing filenames 
        """
        content_or_path = self.id_to_content[audio_id]
        if self.base_content_path:
            content_or_path = os.path.join(
                self.base_content_path, content_or_path
            )  # compatible with s3:// prefix
        content, item_name = self.load_content(audio_id, content_or_path)

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            if self.base_audio_path:
                audio_path = os.path.join(self.base_audio_path, audio_path)
            waveform = self.load_waveform(audio_id, audio_path)
        else:  # inference, only content is available
            waveform = None

        duration = self.load_duration(content, waveform)

        return content, waveform, duration, item_name

    def load_instruction(self) -> torch.Tensor:
        task = self.task
        if self.instruction_idx is None:  # random sample an instruction during training
            num_instruction = self.task_to_num_instruction[task]
            instruction_idx = random.randint(0, num_instruction - 1)
        else:  # use the given instruction index
            instruction_idx = self.instruction_idx - 1

        h5_cache = self.h5_cache if self.use_h5_cache else None
        instruction = read_from_h5(
            f"{task}_{instruction_idx}", self.task_instruction, h5_cache
        )
        return instruction

    def __getitem__(self, index) -> dict[str, Any]:
        audio_id = self.audio_ids[index]
        content, waveform, duration, item_name = self.load_content_waveform(
            audio_id
        )

        if self.id_to_condition:
            condition_path = self.id_to_condition[audio_id]
            condition = self.load_condition(audio_id, condition_path)
        else:
            condition = None

        instruction = self.load_instruction()

        return {
            "item_name": item_name,
            "audio_id": audio_id,
            "content": content,
            "waveform": waveform,
            "condition": condition,
            "duration": duration,
            "task": self.task,
            # "time_aligned": self.time_aligned,
            "instruction": instruction
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

    def load_content(self, audio_id: str,
                     content_or_path: str) -> tuple[Any, str]:
        # text-to-audio / text-to-music, directly use text as the content input
        yid_stem = Path(audio_id).stem
        return content_or_path, f"{yid_stem}_{content_or_path.replace(' ', '_')}"


@dataclass
class TextToMusicDataset(TextToAudioDataset):

    content_col: str = "caption"
    max_duration: float | None = None
    random_crop: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.max_duration is not None:
            self.max_frame_num = int(self.max_duration * self.target_sr)
        else:
            self.max_frame_num = None

    def get_length(self, index: int) -> float:
        orig_duration = super().get_length(index)
        if self.max_duration and self.max_duration < orig_duration:
            return self.max_duration
        return orig_duration

    @property
    def task(self):
        return "text_to_music"

    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        if self.base_content_path:
            content_or_path = os.path.join(
                self.base_content_path, content_or_path
            )
        content, item_name = self.load_content(audio_id, content_or_path)

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            if self.base_audio_path:
                audio_path = os.path.join(self.base_audio_path, audio_path)
            waveform = self.load_waveform(audio_id, audio_path)
            # randomly select a segment
            if self.max_frame_num is not None and len(
                waveform
            ) > self.max_frame_num:
                start_index = random.randint(
                    0,
                    len(waveform) - self.max_frame_num
                ) if self.random_crop else 0
                waveform = waveform[start_index:start_index +
                                    self.max_frame_num]
        else:  # inference, only content is available
            waveform = None

        duration = self.load_duration(content, waveform)
        return content, waveform, duration, item_name


@dataclass(kw_only=True)
class VideoToAudioDataset(AudioGenerationDataset):

    downsampling_ratio: int | None
    content_col: str = "video"

    def __post_init__(self):
        super().__post_init__()
        self.h5_aid_idx_mapping = {}

    def load_content(self, audio_id: str, content_or_path: str):

        if content_or_path not in self.h5_aid_idx_mapping:
            self.h5_aid_idx_mapping[content_or_path] = {}

            with File(content_or_path, "r") as hf:
                aids = hf["audio_id"][()]
                for idx, aid in enumerate(aids):
                    self.h5_aid_idx_mapping[content_or_path][aid.decode()
                                                            ] = idx

        with File(content_or_path, "r") as hf:
            idx = self.h5_aid_idx_mapping[content_or_path][audio_id]
            clip_feature = hf["clip"][idx][()]
            sync_feature = hf["sync"][idx][()]

        yid_stem = Path(audio_id).stem
        content = {
            "clip": clip_feature,
            "sync": sync_feature,
            "duration": 10.0
        }
        return content, yid_stem

    def load_waveform(self, audio_id, audio_path, target_sr=None):
        waveform = super().load_waveform(audio_id, audio_path, target_sr)
        return waveform[:int(self.target_sr * 10.0)]

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        frame_num = int(
            self.target_sr * content["duration"] // self.downsampling_ratio
        )
        duration_value = self.downsampling_ratio / self.target_sr
        duration = np.full(frame_num, duration_value, dtype=np.float32)
        return duration

    @property
    def task(self):
        return "video_to_audio"


@dataclass
class TextToSpeechDataset(AudioGenerationDataset):

    content_col: str = "audio"
    prompt_sr: int = 16000
    min_duration: float = 0.3
    max_duration: float = 30.0

    @property
    def task(self):
        return "text_to_speech"

    def __post_init__(self):
        super().__post_init__()
        self.content_data = {}
        if not isinstance(self.content, list):
            self.content = [self.content]

        self.audio_ids = []
        for jsonl_file in self.content:
            with open(jsonl_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    if self.audio is not None:
                        if (
                            item["duration"] < self.min_duration or
                            item["duration"] > self.max_duration
                        ):
                            continue
                    self.content_data[item["audio_id"]] = item
                    self.audio_ids.append(item["audio_id"])

        if self.max_samples is not None:
            random.shuffle(self.audio_ids)
            self.audio_ids = self.audio_ids[:min(
                len(self.audio_ids), self.max_samples
            )]

    def load_content_waveform(
        self, audio_id: str
    ) -> tuple[Any, torch.Tensor, Sequence[float], str]:

        data_sample = self.content_data[audio_id]
        phoneme = data_sample["phoneme"]
        if isinstance(phoneme, str):
            phoneme = phoneme.split()
        phoneme = [PHONEME2ID[p] for p in phoneme]
        phoneme = np.array(phoneme, dtype=np.long)

        if "phoneme_duration" in data_sample:  # training
            phoneme_duration = data_sample["phoneme_duration"]
            phoneme_duration = np.array(phoneme_duration, dtype=np.float32)
        else:  # inference
            phoneme_duration = np.ones(len(phoneme), dtype=np.float32)

        if "mask_len" in data_sample:  # training
            prompt_duration = data_sample["mask_len"]

            audio_path = self.id_to_audio[audio_id]
            if self.base_audio_path:
                audio_path = os.path.join(self.base_audio_path, audio_path)
            target_waveform = self.load_waveform(audio_id, audio_path)
            prompt_waveform = self.load_waveform(
                audio_id, audio_path, target_sr=self.prompt_sr
            )

            prompt_len_tgt_sr = int(prompt_duration * self.target_sr)
            prompt_len_prm_sr = int(prompt_duration * self.prompt_sr)

            prompt_waveform = prompt_waveform[:prompt_len_prm_sr]
            target_waveform = target_waveform[prompt_len_tgt_sr:]
        else:  # inference
            prompt_audio_path = data_sample["prompt_audio_path"]
            prompt_waveform = self.load_waveform(
                audio_id, prompt_audio_path, target_sr=self.prompt_sr
            )
            target_waveform = None

        content = {
            "prompt_waveform": prompt_waveform,
            "phoneme": phoneme,
        }
        return content, target_waveform, phoneme_duration, audio_id


@dataclass(kw_only=True)
class MidiSingingDataset(AudioGenerationDataset):

    content_col: str = "midi"
    phoneme_set: str | Path
    spk_set: str | Path

    def __post_init__(self):
        super().__post_init__()
        phoneme_list = json.load(open(self.phoneme_set, "r"))
        self.token_encoder = TokenTextEncoder(
            None, vocab_list=phoneme_list, replace_oov=','
        )
        self.spks = json.load(open(self.spk_set, "r"))
        self.spk_map = {spk: i for i, spk in enumerate(self.spks)}

    @property
    def task(self):
        return "singing_voice_synthesis"

    def load_content(self, audio_id: str, content_or_path: str):
        with open(content_or_path, "rb") as file:
            midi = pickle.load(file)[audio_id]
        midi["phoneme"] = self.token_encoder.encode(midi["phoneme"])
        midi["spk"] = self.spk_map[midi["spk"]]
        text = midi["text"]
        return midi, f"{audio_id}_{text}"

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        return np.array(content["phoneme_duration"]).astype(np.float32)


@dataclass(kw_only=True)
class AudioSuperResolutionDataset(AudioGenerationDataset):

    downsampling_ratio: int | None
    content_col: str = "low_sr_audio"
    max_duration: float | None = None
    random_crop: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.max_duration is not None:
            self.max_frame_num = int(self.max_duration * self.target_sr)
        else:
            self.max_frame_num = None

    def get_length(self, index: int) -> float:
        orig_duration = super().get_length(index)
        if self.max_duration and self.max_duration < orig_duration:
            return self.max_duration
        return orig_duration

    @property
    def task(self):
        return "audio_super_resolution"

    def __len__(self) -> int:
        return len(self.audio_ids)

    def load_content(self, audio_id: str, content_or_path: str) -> Any:
        return self.load_waveform(audio_id, content_or_path)

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        if content.dim() == 1:
            frame_num = content.size(0) // self.downsampling_ratio
        else:
            frame_num = content.size(1) // self.downsampling_ratio
        duration_value = self.downsampling_ratio / self.target_sr
        duration = np.full(frame_num, duration_value, dtype=np.float32)
        return duration

    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        if self.base_content_path:
            content_or_path = os.path.join(
                self.base_content_path, content_or_path
            )
        content = self.load_content(audio_id, content_or_path)

        # truncate long audio clip
        if self.max_frame_num is not None and len(
            content
        ) > self.max_frame_num:
            if self.random_crop:
                start_index = random.randint(
                    0,
                    len(content) - self.max_frame_num
                )
            else:
                start_index = 0
            content = content[start_index:start_index + self.max_frame_num]
        else:
            start_index = None

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            if self.base_audio_path:
                audio_path = os.path.join(self.base_audio_path, audio_path)
            waveform = self.load_waveform(audio_id, audio_path)
            if start_index is not None:
                waveform = waveform[start_index:start_index +
                                    self.max_frame_num]
        else:  # inference, only content is available
            waveform = None

        duration = self.load_duration(content, waveform)
        return content, waveform, duration, audio_id


@dataclass(kw_only=True)
class SpeechEnhancementDataset(AudioSuperResolutionDataset):

    content_col: str = "noisy_speech"

    @property
    def task(self):
        return "speech_enhancement"


class AudioGenConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        print(f'\ndatasets:')
        for d in datasets:
            print(f'dataset_name: {d}, len: {len(d)}')
        self.lengths = np.array([len(d) for d in datasets])
        self.cum_sum_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def get_dataset_sample_index(self, idx):
        dataset_idx = np.searchsorted(self.cum_sum_lengths, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sum_lengths[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.get_dataset_sample_index(idx)
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx]

    def get_length(self, idx):
        dataset_idx, sample_idx = self.get_dataset_sample_index(idx)
        dataset = self.datasets[dataset_idx]
        return dataset.get_length(sample_idx)


class TaskGroupedAudioGenConcatDataset(Dataset):
    def __init__(self, datasets: list[AudioGenerationDataset]):
        self.datasets = datasets
        task_to_data_sizes = defaultdict(list)
        self.task_to_datasets = defaultdict(list)
        for dataset in datasets:
            task_to_data_sizes[dataset.task].append(len(dataset))
            self.task_to_datasets[dataset.task].append(dataset)
            # print(f'dataset_name: {dataset}, len: {len(dataset)}')
        self.tasks = list(task_to_data_sizes.keys())

        self.task_to_cum_sum_lengths = {
            task: np.cumsum(sizes)
            for task, sizes in task_to_data_sizes.items()
        }

    def __len__(self):
        return sum(c[-1] for c in self.task_to_cum_sum_lengths.values())

    def get_dataset_sample_index(self, task_idx_tuple):
        task, idx = task_idx_tuple
        cum = self.task_to_cum_sum_lengths[task]
        dataset_idx = np.searchsorted(cum, idx, side='right')
        prev = cum[dataset_idx - 1] if dataset_idx > 0 else 0
        sample_idx = idx - prev
        return dataset_idx, sample_idx

    def __getitem__(self, task_idx_tuple):
        task, idx = task_idx_tuple
        dataset_idx, sample_idx = self.get_dataset_sample_index(task_idx_tuple)
        dataset = self.task_to_datasets[task][dataset_idx]
        return dataset[sample_idx]

    def get_length(self, task_idx_tuple):
        task, idx = task_idx_tuple
        dataset_idx, sample_idx = self.get_dataset_sample_index(task_idx_tuple)
        dataset = self.task_to_datasets[task][dataset_idx]
        return dataset.get_length(sample_idx)


if __name__ == '__main__':

    from tqdm import tqdm
    from accel_hydra.utils.config import load_config_from_cli
    from accel_hydra.utils.data import init_dataloader_from_config

    from utils.config import register_omegaconf_resolvers

    config = load_config_from_cli(
        register_resolver_fn=register_omegaconf_resolvers
    )
    dataloader = init_dataloader_from_config(config["train_dataloader"])

    batch_idx = 0
    for batch in tqdm(dataloader):
        # print(batch["task"])
        batch_idx += 1
        # if batch_idx == 100:
        # break
