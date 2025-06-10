import argparse
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json


def main(save_dir, data_set_path):
    print("Loading dataset...")
    ds = load_dataset(data_set_path)

    save_dir = Path(save_dir)
    for split in ["train", "val", "test"]:
        (save_dir / split).mkdir(parents=True, exist_ok=True)
        with open(save_dir / split / "audio.jsonl", "w") as audio_writer, \
             open(save_dir / split / "caption.jsonl", "w") as caption_writer:
            if split == "val":
                ds_split = "valid"
            else:
                ds_split = split
            for row in tqdm(ds[ds_split], desc=f"Processing {split} split..."):
                audio_writer.write(
                    json.dumps({
                        "audio_id": row['track_id'],
                        "audio": row['path']
                    }) + "\n")
                caption_writer.write(
                    json.dumps({
                        "audio_id": row['track_id'],
                        "caption": row['caption_writing']
                    }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process LP-MusicCaps-MSD Dataset")
    parser.add_argument('--save_dir',
                        type=str,
                        default='./data/msd',
                        help='Directory to save processed data')
    # ref to https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MSD
    parser.add_argument(
        '--data_set_path',
        type=str,
        default=
        "/cpfs_shared/jiahao.mei/code/x_to_audio_generation/tmp/LP-MusicCaps-MSD",
        help='Path to the dataset')

    args = parser.parse_args()

    main(args.save_dir, args.data_set_path)
