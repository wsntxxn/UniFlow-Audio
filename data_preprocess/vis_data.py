from pathlib import Path
from typing import Callable
import argparse
import json
from multiprocessing import Pool

import numpy as np
from h5py import File
from tqdm import tqdm
from moviepy import VideoFileClip
import torchvision
import torch
import pandas as pd
from transformers import CLIPImageProcessor, CLIPVisionModel
from sklearn.model_selection import train_test_split


def process_video(
    video_path: str,
    segment_length: float,
    output_dir: str | Path,
    processed_videos: set,
    processed_video_record: str | Path,
):
    """
    Process a single video file.
    Args:
        video_path: The path of the video file.
        segment_length: The length of each segment in seconds.
        output_dir: The directory to save the processed vis data.
    """
    filename = Path(video_path).stem

    if filename in processed_videos:
        return

    video_clip = VideoFileClip(video_path)
    duration = video_clip.duration
    for start in np.arange(0, duration, segment_length):
        end = min(start + segment_length, duration)
        subclip = video_clip.subclipped(start, end)
        subclip_path = output_dir / f"{filename}_{start:.2f}_{end:.2f}.mp4"
        subclip.write_videofile(str(subclip_path), logger=None)
    video_clip.close()

    with open(processed_video_record, "a") as f:
        f.write(filename + "\n")


def segment_videos(
    video_files: list[str | Path],
    video_segment_dir: str | Path,
    segment_length: float,
    num_workers: int = 4,
):
    """
    Segment the videos into smaller clips.
    Args:
        raw_vis_data_dir: The directory of raw vis data.
        video_segment_dir: The directory of segmented vis data.
        segment_length: The length of each segment in seconds.
        output_dir: The directory to save the processed vis data.
        num_workers: The number of workers to process the data.
    """
    video_segment_dir = Path(video_segment_dir)
    processed_videos = []
    video_segment_dir.mkdir(exist_ok=True, parents=True)
    processed_video_record = video_segment_dir.parent / "processed_videos.txt"
    with open(processed_video_record) as f:
        for line in f.readlines():
            processed_videos.append(line.strip())
    processed_videos = set(processed_videos)
    args = [(
        str(path), segment_length, video_segment_dir, processed_videos,
        processed_video_record
    ) for path in video_files]

    with Pool(num_workers) as pool:
        list(tqdm(pool.starmap(process_video, args), total=len(args)))


def read_video(
    vid_path: str | Path,
    duration: float = 10.0,
    fps: int = 10,
    video_size: tuple = (256, 256),
    transform: Callable = None
):
    try:
        target_length = int(duration * fps)
        video, _, meta = torchvision.io.read_video(
            str(vid_path), start_pts=0, end_pts=duration, pts_unit='sec'
        )
        video_duration = video.shape[0] / meta["video_fps"]

        if video_duration < duration:
            num_frames, height, width, channels = video.shape
            padding_length = int(duration * meta["video_fps"]) - num_frames
            padding = torch.zeros((padding_length, height, width, channels),
                                  dtype=video.dtype)
            video = torch.cat([video, padding], dim=0)

        indices = torch.linspace(0, video.shape[0] - 1,
                                 steps=target_length).long()
        video = video[indices]
        if transform is not None:
            video = transform(video.permute(0, 3, 1, 2))
        return {"video": video, "error": "None"}
    except Exception as e:
        print("Encounter error while reading video:", str(e))
        # If there's an error, return a tensor of zeros
        video = torch.zeros(target_length, 3, *video_size)
        return {"video": video, "error": str(e)}


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        aid_to_video: dict[str, str | Path],
        duration: float = 10.0,
        fps: int = 10,
        video_size: tuple = (256, 256),
        clip_model_name: str = "openai/clip-vit-large-patch14"
    ):
        self.aid_to_video = aid_to_video
        self.aids = list(aid_to_video.keys())
        self.duration = duration
        self.fps = fps
        self.video_size = video_size
        self.transform = torchvision.transforms.Resize(video_size)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            clip_model_name
        )

    def __getitem__(self, index: int):
        audio_id = self.aids[index]
        video_file = self.aid_to_video[audio_id]
        label = "dummy"
        result = read_video(
            video_file, self.duration, self.fps, self.video_size,
            self.transform
        )
        video = self.image_processor(
            images=result["video"], return_tensors="pt"
        )

        return {
            "error": result["error"],
            "video": video,
            "label": label,
            "audio_id": audio_id
        }

    def __len__(self):
        return len(self.aids)


def extract_clip_features(
    aid_to_video: dict[str, str | Path],
    num_workers: int,
    output_dir: str | Path,
    duration: float = 10.0,
    fps: int = 10,
    video_size: tuple = (256, 256),
    clip_model_name: str = "openai/clip-vit-large-patch14"
):
    output_dir = Path(output_dir)
    dataset = VideoDataset(
        aid_to_video, duration, fps, video_size, clip_model_name
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        num_workers=num_workers,
    )
    meta_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = CLIPVisionModel.from_pretrained(clip_model_name
                                                    ).to(device)
    visual_encoder.eval()
    h5_path = output_dir / "clip_features.h5"
    output_dir.mkdir(exist_ok=True, parents=True)
    with File(h5_path, "w") as hf, torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Extracting CLIP features", leave=False
        ):
            batch_size = len(batch["audio_id"])
            num_frames = batch["video"]["pixel_values"].shape[1]
            for k in batch["video"]:
                batch["video"][k] = batch["video"][k].reshape(
                    batch_size * num_frames, *batch["video"][k].shape[2:]
                )
            video_feature = visual_encoder(
                **batch["video"].to(device)
            ).pooler_output

            video_feature = video_feature.reshape(
                batch_size, num_frames, *video_feature.shape[1:]
            )
            video_feature = video_feature.cpu().numpy()

            for sample_idx in range(len(batch["audio_id"])):
                if batch["error"][sample_idx] != "None":
                    continue
                audio_id = batch["audio_id"][sample_idx]
                hf[f"{audio_id}/video"] = video_feature[sample_idx]
                hf[f"{audio_id}/label"] = batch["label"][sample_idx]
                meta_data.append({
                    "audio_id": audio_id,
                    "hdf5_path": str(h5_path.absolute().resolve()),
                })

    feature_df = pd.DataFrame(meta_data)
    feature_df.to_csv(output_dir / f"clip_features.csv", index=False, sep="\t")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_vis_data_dir",
        "-i",
        type=str,
        required=True,
        help="The directory of raw vis data."
    )
    parser.add_argument(
        "--video_segment_dir",
        "-ov",
        type=str,
        required=True,
        help="The directory of segmented vis data."
    )
    parser.add_argument(
        "--segment_length",
        "-sl",
        type=float,
        default=10.0,
        help="The length of each segment in seconds."
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=True,
        help="The directory to save the processed vis data."
    )
    parser.add_argument(
        "--feature_dir",
        "-of",
        type=str,
        required=True,
        help="The directory to save the extracted features."
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        type=int,
        default=4,
        help="The number of workers to process the data."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed for splitting the data."
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=4.0,
        help="The minimum duration of the video to be included in the dataset."
    )

    args = parser.parse_args()

    raw_vis_data_dir = Path(args.raw_vis_data_dir)
    video_segment_dir = Path(args.video_segment_dir)
    segment_length = args.segment_length
    output_dir = Path(args.output_dir)

    video_segment_dir.mkdir(exist_ok=True, parents=True)

    video_files = list(raw_vis_data_dir.glob("*_denoised.mp4"))

    segment_videos(
        video_files, video_segment_dir, segment_length, args.num_workers
    )

    aid_to_video = {}
    for video_file in tqdm(list(Path(video_segment_dir).glob("*.mp4"))):
        try:
            duration = VideoFileClip(str(video_file)).duration
            if duration < args.min_duration:
                continue
            audio_id = video_file.name
            aid_to_video[audio_id] = video_file
        except Exception as e:
            print(f"Error while processing {video_file}: {e}")

    # extract_clip_features(
    #     aid_to_video,
    #     args.num_workers,
    #     args.feature_dir,
    # )

    audio_ids = list(aid_to_video.keys())
    train_audio_ids, test_audio_ids = train_test_split(
        audio_ids, test_size=0.05, random_state=args.seed
    )
    split_audio_ids = {
        "train": train_audio_ids,
        "val": test_audio_ids,
        "test": test_audio_ids,
    }
    h5_path = (Path(args.feature_dir) / "clip_features.h5").resolve().__str__()
    for split in split_audio_ids:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)
        with open(split_dir / "video.jsonl", "w") as video_writer, \
            open(split_dir / "audio.jsonl", "w") as audio_writer:
            for audio_id in split_audio_ids[split]:
                video_writer.write(
                    json.dumps({
                        "audio_id": audio_id,
                        "video": h5_path,
                    }) + "\n"
                )
                audio_writer.write(
                    json.dumps({
                        "audio_id": audio_id,
                        "audio": aid_to_video[audio_id].__str__()
                    }) + "\n"
                )
