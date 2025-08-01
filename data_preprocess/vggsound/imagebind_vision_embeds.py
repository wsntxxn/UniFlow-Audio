import argparse
import torch
import os

import h5py
from tqdm import tqdm

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision.transforms._transforms_video import NormalizeVideo

from evaluation.GMELab.submodules.ImageBind.imagebind.models import imagebind_model
from evaluation.GMELab.submodules.ImageBind.imagebind.models.imagebind_model import ModalityType
from evaluation.GMELab.submodules.ImageBind.imagebind.data import SpatialCrop, get_clip_timepoints
from utils.general import read_jsonl_to_mapping


class VideoBindDataset(Dataset):
    def __init__(
        self,
        aid_to_video,
        device,
        clip_duration=2,
        clips_per_video=5,
    ):
        self.aid_to_video = aid_to_video
        self.aids = list(aid_to_video.keys())
        self.device = device
        self.clip_duration = clip_duration
        self.clips_per_video = clips_per_video
        self.video_transform = transforms.Compose([
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # [0,255] → [0,1]
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration,
            clips_per_video=clips_per_video,
        )
        self.frame_sampler = pv_transforms.UniformTemporalSubsample(
            num_samples=clip_duration
        )

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, idx):
        audio_id = self.aids[idx]
        video_path = self.aid_to_video[audio_id]
        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
        )

        all_clips_timepoints = get_clip_timepoints(
            self.clip_sampler, video.duration
        )

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = self.frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [self.video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)
        return audio_id, all_video


def main(
    jsonl_path: str,
    output_path: str,
    batch_size: int = 4,
    num_workers: int = 8
):
    aid_to_video = read_jsonl_to_mapping(jsonl_path, "audio_id", "video")

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True, )
    model.eval()
    model.to(device)

    video_dataset = VideoBindDataset(
        aid_to_video,
        device,
    )

    video_dataloader = DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    with h5py.File(output_path, 'w') as f_vis:
        for audio_ids, video_data in tqdm(
            video_dataloader,
            total=len(video_dataloader),
            desc='Saving embeddings'
        ):
            video_data = video_data.to(device)

            with torch.no_grad():
                embeddings = model({ModalityType.VISION: video_data})
                embeddings = embeddings[ModalityType.VISION].cpu().numpy()

            for aid, embed in zip(audio_ids, embeddings):
                f_vis[aid] = embed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl_path",
        "-j",
        type=str,
        required=True,
        help="Path to video jsonl file"
    )

    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="Path to output h5 file"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=4,
        help="Batch size for dataloader"
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        type=int,
        default=8,
        help="Number of workers for dataloader"
    )

    args = parser.parse_args()
    main(args.jsonl_path, args.output_path, args.batch_size, args.num_workers)
