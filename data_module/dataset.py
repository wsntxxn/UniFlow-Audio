from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
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
import random

from utils.diffsinger_utilities import norm_interp_f0, TokenTextEncoder
from utils.general import read_jsonl_to_mapping
from constants import TIME_ALIGNED_TASKS, NON_TIME_ALIGNED_TASKS


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
        if self.task in TIME_ALIGNED_TASKS:
            self.is_time_aligned = True
        elif self.task in NON_TIME_ALIGNED_TASKS:
            self.is_time_aligned = False
        else:
            raise Exception(
                f"Time align property of {self.task} is not defined!"
            )


@dataclass(kw_only=True)
class AudioWaveformDataset(HDF5DatasetMixin):

    target_sr: int | None = None
    use_h5_cache: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.h5_src_sr_map = {}

    def load_waveform(self, audio_id: str, audio_path: str):
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
        else:
            try:
                waveform, orig_sr = torchaudio.load(audio_path)

            except Exception:
                print(f"Failed to load audio from {audio_path}")
                with open('./broken_audio_list.txt', 'a') as f:
                    f.write(audio_id + ',' + audio_path + '\n')
                return torch.zeros([100], dtype=torch.float32)
            # average multi-channel to single-channel
            waveform = waveform.mean(0)

        if self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=self.target_sr
            )
        return waveform


@dataclass
class AudioGenerationDataset(AudioWaveformDataset, TaskMixin):

    content: str | Path
    audio: str | Path | None = None
    condition: str | Path | None = None

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
        self.base_content_path = Path(
            self.base_content_path
        ) if self.base_content_path else None

        id_col_in_audio = self.id_col_in_audio or self.id_col
        if self.audio:
            self.id_to_audio = read_jsonl_to_mapping(
                self.audio, id_col_in_audio, self.audio_col
            )
        else:
            self.id_to_audio = None
        # id_to_audio: {'id1': '<audio path1>', 'id2': '<audio path2>'}
        self.base_audio_path = Path(
            self.base_audio_path
        ) if self.base_audio_path else None

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
            content_or_path = str(self.base_content_path / content_or_path)
        content, item_name = self.load_content(audio_id, content_or_path)

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            if self.base_audio_path:
                audio_path = str(self.base_audio_path / audio_path)
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
            "is_time_aligned": self.is_time_aligned,
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

    @property
    def task(self):
        return "text_to_music"

    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        if self.base_content_path:
            content_or_path = str(self.base_content_path / content_or_path)
        content, item_name = self.load_content(audio_id, content_or_path)

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            if self.base_audio_path:
                audio_path = str(self.base_audio_path / audio_path)
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


@dataclass
class VideoToAudioDataset(AudioGenerationDataset):

    content_col: str = "video"
    video_fps: int = 10

    def load_content(self, audio_id: str, content_or_path: str):
        video_path = self.id_to_content[audio_id]

        if video_path.endswith(".hdf5") or video_path.endswith(".h5"):
            with File(video_path, "r") as hf:
                video_feature = hf[f"{audio_id}/video"][()]
                label: bytes = hf[f"{audio_id}/label"][()]
                label = label.decode()

        else:
            raise NotImplementedError(
                "video feature must be load by h5 file !"
            )

        yid_stem = Path(audio_id).stem
        return video_feature, f"{yid_stem}_{label}"

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        clip_duration = content.shape[0] / self.video_fps
        frame_num = content.shape[0]
        duration_value = clip_duration / frame_num
        duration = np.full(frame_num, duration_value, dtype=np.float32)
        return duration

    @property
    def task(self):
        return "video_to_audio"


@dataclass
class TextToSpeechDataset(AudioGenerationDataset):

    content_col: str = "audio"

    @property
    def task(self):
        return "text_to_speech"

    def load_content(self, audio_id, content_or_path):
        with File(content_or_path, "r") as hf:
            phoneme = hf["phoneme"][audio_id][()]
            phoneme_duration = hf["phoneme_duration"][audio_id][()]

            if "xvector" in hf.keys():
                spk = hf["xvector"][audio_id][()]
            else:
                spk = None

        content = {
            "phoneme": phoneme,
            "phoneme_duration": phoneme_duration,
            "spk": spk,
        }
        return content, audio_id

    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        return content["phoneme_duration"].astype(np.float32)


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


@dataclass(kw_only=True)
class AudioSuperResolutionDataset(AudioGenerationDataset):

    downsampling_ratio: int | None
    content_col: str = "caption"
    max_duration: float | None = None
    random_crop: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.max_duration is not None:
            self.max_frame_num = int(self.max_duration * self.target_sr)
        else:
            self.max_frame_num = None

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
            content_or_path = str(self.base_content_path / content_or_path)
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
                audio_path = str(self.base_audio_path / audio_path)
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

    id_col: str = "UUID"
    content_col: str = "InputPath"
    audio_col: str = "WavPath"

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

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cum_sum_lengths, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sum_lengths[dataset_idx - 1]
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx]


class TaskGroupedAudioGenConcatDataset(Dataset):
    def __init__(self, datasets: list[AudioGenerationDataset]):
        self.datasets = datasets
        task_to_data_sizes = defaultdict(list)
        self.task_to_datasets = defaultdict(list)
        print(f'\n train datasets \n')
        for dataset in datasets:
            task_to_data_sizes[dataset.task].append(len(dataset))
            self.task_to_datasets[dataset.task].append(dataset)
            print(f'dataset_name:{dataset},len:{len(dataset)}')
        self.tasks = list(task_to_data_sizes.keys())

        self.task_to_cum_sum_lengths = {
            task: np.cumsum(sizes)
            for task, sizes in task_to_data_sizes.items()
        }

    def __len__(self):
        return sum(c[-1] for c in self.task_to_cum_sum_lengths.values())

    def __getitem__(self, task_idx_tuple):
        task, idx = task_idx_tuple
        cum = self.task_to_cum_sum_lengths[task]
        dataset_idx = np.searchsorted(cum, idx, side='right')
        prev = cum[dataset_idx - 1] if dataset_idx > 0 else 0
        sample_idx = idx - prev
        dataset = self.task_to_datasets[task][dataset_idx]
        return dataset[sample_idx]


if __name__ == '__main__':

    from tqdm import tqdm
    from data_module.sampler import TaskIteratingSampler, TaskGroupedIteratingBatchSampler, TaskGroupedSequentialBatchSampler
    from data_module.collate_function import PaddingCollate, PaddingCollateWithAnyContent

    # dataset = TaskGroupedAudioGenConcatDataset(
    dataset = AudioGenConcatDataset(
        datasets=[
            # TextToSpeechDataset(
            #     content="data/libritts/train/phoneme.jsonl",
            #     audio="data/libritts/train/audio.jsonl",
            #     base_content_path=
            #     "/cpfs04/user/xuxuenan/workspace/x_to_audio_generation",
            #     base_audio_path="/cpfs02/shared/speechllm/",
            #     target_sr=24000,
            #     task_instruction="./data/instructions/t5_embeddings.h5",
            # ),
            # SpeechEnhancementDataset(
            #     content=
            #     "/oss-speechllm-a100/xuxuenan/speech_enhancement/Libritts+Wham/train/metadata_caption.jsonl",
            #     audio=
            #     "/oss-speechllm-a100/xuxuenan/speech_enhancement/Libritts+Wham/train/metadata_audio.jsonl",
            #     base_content_path=
            #     "/oss-speechllm-a100/xuxuenan/speech_enhancement/Libritts+Wham",
            #     base_audio_path=
            #     "/oss-speechllm-a100/xuxuenan/speech_enhancement/Libritts+Wham",
            #     downsampling_ratio=480,
            #     target_sr=24000,
            #     task_instruction="./data/instructions/t5_embeddings.h5",
            #     max_duration=5.0,
            # ),
            # TextToAudioDataset(
            #     content="./data/audiocaps_v2/test/caption.jsonl",
            #     audio="./data/audiocaps_v2/test/audio.jsonl",
            #     task_instruction="./data/instructions/t5_embeddings.h5",
            #     instruction_idx=1,
            #     target_sr=24000
            # ),
            # MidiSingingDataset(
            #     content="./data/m4singer/val/midi.jsonl",
            #     audio="./data/m4singer/val/audio.jsonl",
            #     phoneme_set="./data/m4singer/phone_set.json",
            #     spk_set="./data/m4singer/spk_set.json",
            #     target_sr=24000,
            #     task_instruction="./data/instructions/t5_embeddings.h5",
            #     instruction_idx=4
            # ),
            # VideoToAudioDataset(
            #     content="./data/vggsound/clip/train/content.jsonl",
            #     audio="./data/vggsound/clip/train/audio.jsonl",
            #     video_fps=10,
            #     target_sr=24000,
            #     task_instruction="./data/instructions/t5_embeddings.h5",
            # ),
            TextToMusicDataset(
                content="data/msd/train/caption.jsonl",
                audio="data/msd/train/audio.jsonl",
                base_audio_path=
                "/cpfs02/shared/speechllm/million_song_dataset/files/MSD/songs/",
                max_duration=10.0,
                target_sr=24000,
                task_instruction="./data/instructions/t5_embeddings.h5",
            )
        ]
    )
    # sampler = TaskIteratingSampler(dataset, shuffle=True)
    # batch_sampler = TaskGroupedSequentialBatchSampler(
    #     dataset, batch_size=16, shuffle=False
    # )

    # collate_fn = PaddingCollateWithAnyContent(
    #     pad_keys=["waveform", "duration", "instruction"],
    #     content_pad_keys=[
    #         "phoneme", "phoneme_duration", "midi", "midi_duration", "is_slur",
    #         "frames"
    #     ],
    #     content_torchify_keys=["spk"]
    # )
    collate_fn = PaddingCollate(
        pad_keys=["waveform", "duration", "instruction"],
        torchify_keys="is_time_aligned"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=8,
        # sampler=sampler,
        batch_size=32,
        # batch_sampler=batch_sampler
    )

    batch_idx = 0
    for batch in tqdm(dataloader):
        print(batch["task"])
        batch_idx += 1
        # if batch_idx == 100:
        # break
