import logging
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import av
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import snapshot_download
from open_clip import create_model_from_pretrained
from torchvision.transforms import v2
from torchvision.transforms import Normalize

from .synchformer import Synchformer

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

log = logging.getLogger()


@dataclass
class ModelConfig:
    clip_ckpt: str | Path
    synchformer_ckpt: str | Path

    clip_frame_rate: int = 8
    sync_frame_rate: int = 25
    sync_num_frames_per_segment: int = 16
    sync_step_size: int = 8
    sync_downsample_rate: int = 2


@dataclass
class ImageInfo:
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    original_frame: np.ndarray | None

    @property
    def height(self):
        return self.original_frame.shape[0]

    @property
    def width(self):
        return self.original_frame.shape[1]


@dataclass
class VideoInfo:
    duration_sec: float
    fps: Fraction
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    all_frames: list[np.ndarray] | None

    @property
    def height(self):
        return self.all_frames[0].shape[0]

    @property
    def width(self):
        return self.all_frames[0].shape[1]

    @classmethod
    def from_image_info(
        cls, image_info: ImageInfo, duration_sec: float, fps: Fraction
    ) -> 'VideoInfo':
        num_frames = int(duration_sec * fps)
        all_frames = [image_info.original_frame] * num_frames
        return cls(
            duration_sec=duration_sec,
            fps=fps,
            clip_frames=image_info.clip_frames,
            sync_frames=image_info.sync_frames,
            all_frames=all_frames
        )


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):
    def __init__(
        self,
        *,
        clip_ckpt: str,
        synchformer_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.clip_model = create_model_from_pretrained(
            f'local-dir:{clip_ckpt}', return_transform=False
        )
        self.clip_preprocess = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.clip_model = patch_clip(self.clip_model)

        self.synchformer = Synchformer()
        self.synchformer.load_state_dict(
            torch.load(
                synchformer_ckpt, weights_only=True, map_location='cpu'
            )
        )

        self.tokenizer = open_clip.get_tokenizer(
            'ViT-H-14-378-quickgelu'
        )  # same as 'ViT-H-14'

    @torch.inference_mode()
    def encode_video_with_clip(
        self, x: torch.Tensor, batch_size: int = -1
    ) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        batch_size = int(batch_size)
        for i in range(0, int(b * t), batch_size):
            outputs.append(
                self.clip_model.encode_image(
                    x[i:i + batch_size], normalize=True
                )
            )
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_sync(
        self, x: torch.Tensor, batch_size: int = -1
    ) -> torch.Tensor:
        assert self.synchformer is not None, 'Synchformer is not loaded'
        # x: (B, T, C, H, W) H/W: 384

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def read_frames(
    video_path: Path, list_of_fps: list[float], start_sec: float,
    end_sec: float, need_all_frames: bool
) -> tuple[list[np.ndarray], list[np.ndarray], Fraction]:
    output_frames = [[] for _ in list_of_fps]
    next_frame_time_for_each_fps = [0.0 for _ in list_of_fps]
    time_delta_for_each_fps = [1 / fps for fps in list_of_fps]
    all_frames = []

    # container = av.open(video_path)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = stream.guessed_rate
        stream.thread_type = 'AUTO'
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = frame.time
                if frame_time < start_sec:
                    continue
                if frame_time > end_sec:
                    break

                frame_np = None
                if need_all_frames:
                    frame_np = frame.to_ndarray(format='rgb24')
                    all_frames.append(frame_np)

                for i, _ in enumerate(list_of_fps):
                    this_time = frame_time
                    while this_time >= next_frame_time_for_each_fps[i]:
                        if frame_np is None:
                            frame_np = frame.to_ndarray(format='rgb24')

                        output_frames[i].append(frame_np)
                        next_frame_time_for_each_fps[
                            i] += time_delta_for_each_fps[i]

    output_frames = [np.stack(frames) for frames in output_frames]
    return output_frames, all_frames, fps


def load_video(
    video_path: Path,
    duration_sec: float,
    load_all_frames: bool = True
) -> VideoInfo:

    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE),
                  interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if not isinstance(video_path, Path) and not isinstance(video_path, str):
        # for gradio>6.0, video_path is a namespace
        video_path = Path(video_path.value)
    output_frames, all_frames, orig_fps = read_frames(
        video_path,
        list_of_fps=[_CLIP_FPS, _SYNC_FPS],
        start_sec=0,
        end_sec=duration_sec,
        need_all_frames=load_all_frames
    )

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

    if clip_length_sec < duration_sec:
        log.warning(
            f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}'
        )
        log.warning(f'Truncating to {clip_length_sec:.2f} sec')
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        log.warning(
            f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}'
        )
        log.warning(f'Truncating to {sync_length_sec:.2f} sec')
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
    sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=orig_fps,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        all_frames=all_frames if load_all_frames else None,
    )
    return video_info


def get_video_feature(
    video_path: str,
    feature_utils,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    duration_sec: float = 10.0,
):
    video_info = load_video(video_path, duration_sec)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames

    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)

    device = feature_utils.device
    dtype = feature_utils.dtype
    clip_video = clip_frames.to(device, dtype, non_blocking=True)
    clip_features = feature_utils.encode_video_with_clip(
        clip_video, batch_size=clip_batch_size_multiplier
    )

    sync_video = sync_frames.to(device, dtype, non_blocking=True)
    sync_features = feature_utils.encode_video_with_sync(
        sync_video, batch_size=sync_batch_size_multiplier
    )

    return {
        "clip": clip_features[0],
        "sync": sync_features[0],
        "duration": video_info.duration_sec
    }


def init_feature_extractor():
    if "APPLE_CLIP_CKPT_PATH" in os.environ and os.path.exists(
        os.environ["APPLE_CLIP_CKPT_PATH"]
    ):
        clip_ckpt = os.environ["APPLE_CLIP_CKPT_PATH"]
    else:
        clip_ckpt = snapshot_download("apple/DFN5B-CLIP-ViT-H-14-378")

    if "SYNCHFORMER_CKPT_PATH" in os.environ and os.path.exists(
        os.environ["SYNCHFORMER_CKPT_PATH"]
    ):
        synchformer_ckpt = os.environ["SYNCHFORMER_CKPT_PATH"]
    else:
        synchformer_ckpt = Path(
            snapshot_download("hkchengrex/MMAudio")
        ) / "ext_weights/synchformer_state_dict.pth"
    model_config = ModelConfig(
        clip_ckpt=clip_ckpt, synchformer_ckpt=synchformer_ckpt
    )
    feature_utils = FeaturesUtils(
        clip_ckpt=model_config.clip_ckpt,
        synchformer_ckpt=model_config.synchformer_ckpt,
    )
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32
    feature_utils = feature_utils.to(device, dtype).eval()

    return feature_utils
