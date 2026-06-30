from pathlib import Path
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import h5py
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

from ...submodules.ImageBind.imagebind.data import (
    get_clip_timepoints,
    waveform2melspec,
)
from ...submodules.ImageBind.imagebind.models import imagebind_model
from ...submodules.ImageBind.imagebind.models.imagebind_model import ModalityType


@dataclass
class ImageBindScore:
    device: str = "cuda:0"
    afps: int = 16_000
    get_diagonal_scores: bool = False


BATCH_SIZE = 1


class AudioDataset(Dataset):
    def __init__(
        self,
        aid_to_audio: dict[str, str | Path],
        device: str,
        num_mel_bins: int = 128,
        target_length: int = 204,
        sample_rate: int = 16000,
        clip_duration: float = 2,
        clips_per_video: int = 3,
        mean: float = -4.268,
        std: float = 9.138,
    ):
        self.aids = list(aid_to_audio.keys())
        self.aid_to_audio = aid_to_audio
        self.device = device
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clips_per_video = clips_per_video
        self.mean = mean
        self.std = std

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration,
            clips_per_video=clips_per_video,
        )

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, idx):
        audio_id = self.aids[idx]
        audio_path = self.aid_to_audio[audio_id]

        waveform, sr = torchaudio.load(audio_path)

        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        all_clips_timepoints = get_clip_timepoints(
            self.clip_sampler,
            waveform.size(1) / self.sample_rate
        )

        all_clips = []
        for t0, t1 in all_clips_timepoints:
            clip = waveform[:,
                            int(t0 *
                                self.sample_rate):int(t1 * self.sample_rate)]
            mel = waveform2melspec(
                clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_clips.append(mel)

        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        all_clips = torch.stack([normalize(c) for c in all_clips], dim=0)

        return audio_id, all_clips  # shape: [clips_per_video, 1, mel_bins, time_frames]


def calculate_imagebind_score(
    aid_to_audio: dict[str, str | Path],
    device: str,
    vision_embeds: str,
    get_diagonal_scores: bool = True,
    afps: int = 16000,
    verbose: bool = False,
    num_workers: int = 8,
):
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    # load model
    if distributed and rank == 0:
        model = imagebind_model.imagebind_huge(pretrained=True)
        dist.barrier()
    elif distributed:
        dist.barrier()
        model = imagebind_model.imagebind_huge(pretrained=True)
    else:
        model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.device(device).type == "cuda":
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    cos_sim = torch.nn.CosineSimilarity(dim=1)

    audio_dataset = AudioDataset(
        aid_to_audio,
        device,
    )
    sampler = DistributedSampler(
        audio_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if distributed else None
    audio_loader = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
    )

    scores = {}
    # run model inference
    with h5py.File(vision_embeds, 'r') as f_vis:
        assert len(f_vis.keys()) == len(aid_to_audio)
        for batch in tqdm(
            audio_loader,
            desc='Compute ImageBind score',
            disable=distributed and rank != 0,
        ):
            keys, audio_data = batch
            audio_data = audio_data.to(device)

            inputs = {
                ModalityType.AUDIO: audio_data,
            }

            with torch.no_grad():
                embeddings = model(inputs)

            aud_embs = embeddings[ModalityType.AUDIO]
            for key, aud_emb in zip(keys, aud_embs):
                vis_emb = torch.as_tensor(f_vis[key][()]).to(device)
                sim_score = cos_sim(vis_emb.unsqueeze(0), aud_emb.unsqueeze(0))
                sim_score = sim_score.item()
                scores[key] = sim_score

    if distributed:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, scores)
        scores = {}
        for part in gathered:
            scores.update(part or {})

    score = sum(scores.values()) / len(scores)
    if verbose:
        print("ImageBind score:", score)
    return float(score)
