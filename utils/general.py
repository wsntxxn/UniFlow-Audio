import json
import re
from typing import Union, Dict
from pathlib import Path
import os

import torch
import numpy as np
from pathlib import Path
import torchvision
from transformers import CLIPImageProcessor, CLIPVisionModel


def read_video_frames(
    video_path, duration=10.0, fps=10, video_size=(256, 256)
):

    target_length = int(duration * fps)
    try:
        video, _, meta = torchvision.io.read_video(
            str(video_path), start_pts=0, end_pts=duration, pts_unit='sec'
        )
        video_duration = video.shape[0] / meta["video_fps"]

        # 如果视频不足duration秒，则补零帧
        if video_duration < duration:
            num_frames, height, width, channels = video.shape
            padding_length = int(duration * meta["video_fps"]) - num_frames
            padding = torch.zeros((padding_length, height, width, channels),
                                  dtype=video.dtype)
            video = torch.cat([video, padding], dim=0)

        # 均匀采样为 target_length 帧
        indices = torch.linspace(0, video.shape[0] - 1,
                                 steps=target_length).long()
        video = video[indices]
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        resize = torchvision.transforms.Resize(video_size)
        video = resize(video)
        return video
    except Exception as e:
        print(f"❌ error reading video: {e}")
        return torch.zeros(target_length, 3, *video_size)


def extract_clip_feature_single_video(
    video_path,
    duration=10.0,
    fps=10,
    video_size=(256, 256),
    clip_model_name="openai/clip-vit-large-patch14"
):

    # 加载 CLIP 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device)
    vision_model.eval()

    # 读取帧
    video = read_video_frames(video_path, duration, fps, video_size)
    pixel_values = image_processor(images=video, return_tensors="pt"
                                  ).pixel_values.to(device)

    # 前向提取特征
    with torch.no_grad():
        outputs = vision_model(pixel_values)
        features = outputs.pooler_output  # [num_frames, 1024]

    features_np = features.cpu().numpy()
    print(f"✅ clip feature extracted: {features_np.shape}")  # (10, 1024)

    return features_np


def extract_clip_feature_from_single_video(
    video_path, save_path='./data/tmp_video_feature.npy'
):
    features = extract_clip_feature_single_video(Path(video_path))
    np.save(save_path, features)
    return features, save_path


MAX_FILE_NAME_LENGTH = 100
TASK2DATASET_CLASS = {
    't2a': "TextToAudioDataset",
    't2m': "TextToMusicDataset",
    'se': "SpeechEnhancementDataset",
    'sr': "AudioSuperResolutionDataset",
    'v2a': "VideoToAudioDataset",
    'svs': "MidiSingingDataset",
    'tts': "TextToSpeechDataset"
}


def read_jsonl_to_mapping(
    jsonl_file: Union[str, Path],
    key_col: str,
    value_col: str,
    base_path=None
) -> Dict[str, str]:
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
            if base_path:
                value = os.path.join(base_path, value)
            mapping[key] = value
    return mapping


def sanitize_filename(name: str, max_len: int = MAX_FILE_NAME_LENGTH) -> str:
    """
    Clean and truncate a string to make it a valid and safe filename.
    """
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = name.replace('/', '_')
    max_len = min(len(name), max_len)
    return name[:max_len]


def transform_gen_fn_to_id(audio_file: Path, task: str) -> str:
    if task == "svs":
        audio_id = audio_file.stem.split("_")[0]
    elif task == "sr":
        audio_id = audio_file.stem
    elif task == "tta":
        audio_id = audio_file.stem[:11]
        # audio_id = audio_file.stem[:12] + '.wav'
    elif task == "ttm":
        audio_id = audio_file.stem[:11]
        # audio_id = audio_file.stem[:12] + '.wav'
    elif task == "v2a":
        audio_id = audio_file.stem.rsplit("_", 1)[0] + ".mp4"
    else:
        audio_id = audio_file.stem
    return audio_id


def audio_dir_to_mapping(audio_dir: str | Path, task: str) -> dict:
    mapping = {}
    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.iterdir())
    for audio_file in audio_files:
        audio_id = transform_gen_fn_to_id(audio_file, task)
        mapping[audio_id] = str(audio_file.resolve())
    return mapping
