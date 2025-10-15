from pathlib import Path
import os
from typing import Callable
import tempfile

import numpy as np
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx import AudioLoop
import torch
import torchvision


def merge_audio_video(
    audio: str | Path | np.ndarray,
    video_path: str | Path,
    target_path: str | Path,
    backend: str = "moviepy",
    logging: bool = False,
    audio_fps: int | None = None
):
    """
    Merge audio and video into a single file.

    Args:
        audio_path (str | Path): Path to the audio file.
        video_path (str | Path): Path to the video file.
        target_path (str | Path): Path to the target file.
        backend (str, optional): The backend to use for merging. Defaults to "moviepy".
    """
    assert backend in [
        "moviepy", "ffmpeg"
    ], "Backend should be moviepy or ffmpeg"
    if backend == "moviepy":
        video = VideoFileClip(video_path.__str__())
        video = video.without_audio()
        if isinstance(audio, np.ndarray):
            assert audio_fps is not None
            # write to a temp file, then use AudioFileClip to load
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_wav:
                sf.write(tmp_wav.name, audio, samplerate=audio_fps)
            audio = AudioFileClip(tmp_wav.name)
        else:
            audio = AudioFileClip(audio.__str__())
            tmp_wav = None

        video = video.with_audio(audio)

        target_path = Path(target_path)
        video.write_videofile(
            target_path,
            logger=None if not logging else "bar",
            threads=8,
            preset="ultrafast",
            ffmpeg_params=["-crf", "23"]
        )
        if tmp_wav:
            os.remove(tmp_wav.name)
    else:
        logging_arg = "" if logging else "-loglevel quiet"
        command = f"ffmpeg {logging_arg} -i '{video_path.__str__()}' -i '{audio.__str__()}' -c:v copy " \
                  f"-c:a copy -map 0:v:0 -map 1:a:0 '{target_path.__str__()}'"
        os.system(command)


def read_video_frames(
    video_path: str,
    duration: float | None = 10.0,
    fps: int = 10,
    video_size: tuple[int] = (256, 256),
    resize_transform: Callable | None = None,
):
    try:
        video, _, meta = torchvision.io.read_video(
            str(video_path), start_pts=0, end_pts=duration, pts_unit='sec'
        )
        video_duration = video.shape[0] / meta["video_fps"]

        if duration and video_duration < duration:
            num_frames, height, width, channels = video.shape
            padding_length = int(duration * meta["video_fps"]) - num_frames
            padding = torch.zeros((padding_length, height, width, channels),
                                  dtype=video.dtype)
            video = torch.cat([video, padding], dim=0)
            target_length = int(duration * fps)
        else:
            target_length = int(video_duration * fps)

        indices = torch.linspace(0, video.shape[0] - 1,
                                 steps=target_length).long()
        video = video[indices]
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        if resize_transform is None:
            resize_transform = torchvision.transforms.Resize(video_size)
        video = resize_transform(video)
        return video
    except Exception as e:
        print(f"error reading video {video_path}: {e}")
        assert duration is not None
        target_length = int(duration * fps)
        return torch.zeros(target_length, 3, *video_size)
