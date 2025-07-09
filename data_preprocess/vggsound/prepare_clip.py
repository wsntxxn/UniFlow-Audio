from pathlib import Path
import argparse

import pandas as pd
from tqdm import tqdm
import h5py
import torch
import torchvision
import torchvision.transforms as T
from transformers import CLIPImageProcessor, CLIPVisionModel

FILE_LIST_DIR = Path("data/vggsound/csv_parts")
OUTPUT_DIR = Path("data/vggsound/features")
VGGSOUND_VIDEO_DIR = Path("/cpfs02/shared/speechllm/vggsound/video")
VGGSOUND_LABEL_PATH = "/cpfs02/shared/speechllm/vggsound/label.csv"
MODEL_NAME = "openai/clip-vit-large-patch14"

DURATION = 10.0
VIDEO_FPS = 10
TARGET_LENGTH = int(VIDEO_FPS * DURATION)
VIDEO_SIZE = (256, 256)
RESIZE_TRANSFORM = T.Resize(VIDEO_SIZE)

BATCH_SIZE = 4
NUM_WORKERS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 20000

parser = argparse.ArgumentParser(description="Prepare VGGSound features")
parser.add_argument("--part_index", type=int, required=True)
args = parser.parse_args()


def build_label_mapping():
    df = pd.read_csv(VGGSOUND_LABEL_PATH, header=None)
    mapping = {}
    for _, row in df.iterrows():
        mapping[f"{row[0]}_{row[1]}.mp4"] = row[2]
    return mapping


VGGSOUND_LABEL_MAPPING = build_label_mapping()


def read_video(vid_path):
    try:
        video, _, meta = torchvision.io.read_video(
            str(vid_path), start_pts=0, end_pts=DURATION, pts_unit='sec'
        )
        video_duration = video.shape[0] / meta["video_fps"]

        if video_duration < DURATION:
            num_frames, height, width, channels = video.shape
            padding_length = int(DURATION * meta["video_fps"]) - num_frames
            padding = torch.zeros((padding_length, height, width, channels),
                                  dtype=video.dtype)
            video = torch.cat([video, padding], dim=0)

        indices = torch.linspace(0, video.shape[0] - 1,
                                 steps=TARGET_LENGTH).long()
        video = video[indices]
        video = RESIZE_TRANSFORM(video.permute(0, 3, 1, 2))
        # video_feature = get_clip_embeds(video)
        return {"video": video, "error": "None"}
    except Exception as e:
        print("Encounter error while reading video:", str(e))
        # If there's an error, return a tensor of zeros
        video = torch.zeros(TARGET_LENGTH, 3, *VIDEO_SIZE)
        return {"video": video, "error": str(e)}


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.image_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        vid_path = VGGSOUND_VIDEO_DIR / row["audio_id"]
        label = VGGSOUND_LABEL_MAPPING[row["audio_id"]]
        result = read_video(vid_path)
        video = self.image_processor(
            images=result["video"], return_tensors="pt"
        )

        return {
            "error": result["error"],
            "video": video,
            "label": label,
            "audio_id": row["audio_id"]
        }

    def __len__(self):
        return len(self.dataframe)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        FILE_LIST_DIR / f"part_{args.part_index:02d}.csv", sep="\t"
    )

    visual_encoder = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)
    visual_encoder.eval()

    h5_path = OUTPUT_DIR / f"part_{args.part_index:02d}.h5"

    dataset = VideoDataset(df)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        num_workers=NUM_WORKERS,
    )
    meta_data = []
    with h5py.File(h5_path, "w") as hf, torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Chunk {args.part_index}", leave=False
        ):
            batch_size = len(batch["audio_id"])
            num_frames = batch["video"]["pixel_values"].shape[1]
            for k in batch["video"]:
                batch["video"][k] = batch["video"][k].reshape(
                    batch_size * num_frames, *batch["video"][k].shape[2:]
                )
            video_feature = visual_encoder(
                **batch["video"].to(DEVICE)
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

        pd.DataFrame(meta_data).to_csv(
            OUTPUT_DIR / f"part_{args.part_index:02d}.csv",
            index=False,
            sep="\t"
        )


if __name__ == "__main__":
    main()
