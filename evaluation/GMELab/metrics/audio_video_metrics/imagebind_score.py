from pathlib import Path
from math import floor

import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from evaluation.GMELab.submodules.ImageBind.imagebind.data import (
    load_and_transform_video_data,
    load_and_transform_audio_data,
)
from evaluation.GMELab.submodules.ImageBind.imagebind.models import imagebind_model
from evaluation.GMELab.submodules.ImageBind.imagebind.models.imagebind_model import ModalityType


@dataclass
class ImageBindScore:
    device: str = "cuda:0"
    afps: int = 16_000
    get_diagonal_scores: bool = False


BATCH_SIZE = 1


def calculate_imagebind_score(
    video_dir: Path,
    device: str,
    get_diagonal_scores: bool = True,
    afps: int = 16000,
    verbose: bool = False,
):
    # get videos
    all_videos = list(video_dir.glob("*.mp4"))

    # load model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    running_score = 0
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    # run model inference
    for i in tqdm(
        range(0, len(all_videos), BATCH_SIZE),
        desc="Calculating ImageBind score"
    ):
        # load video and audio data
        try:
            video_data = load_and_transform_video_data(
                all_videos[i:i + BATCH_SIZE],
                device,
                sample_rate=afps,
            )
            audio_data = load_and_transform_audio_data(
                all_videos[i:i + BATCH_SIZE],
                device,
                sample_rate=afps,
            )
        except Exception as e:
            print(e)
            continue
        inputs = {
            ModalityType.AUDIO: audio_data,
            ModalityType.VISION: video_data,
        }

        with torch.no_grad():
            embeddings = model(inputs)

        sim_scores = cos_sim(
            embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]
        )
        sim_scores = sim_scores.cpu().numpy()
        running_score += np.sum(sim_scores)

    score = running_score / len(all_videos)
    if verbose:
        print("ImageBind score:", score)
    return float(score)


if __name__ == '__main__':
    image_bind_score = calculate_imagebind_score(
        Path(
            "/hpc_stor03/sjtu_home/yaoyun.zhang/project/x_to_audio_generation/evaluation/fad-test/samples/gen-video-5.12s-25fps-16000hz"
        ), "cuda"
    )
    print("image bind score:", image_bind_score)
