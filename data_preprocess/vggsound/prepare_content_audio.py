from pathlib import Path
import argparse
import pandas as pd
import json
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vggsound_dir", type=str)
    parser.add_argument(
        "--feature_csv_path",
        type=str,
        default="./data/vggsound/clip_feature.csv"
    )
    parser.add_argument(
        "--target_vggsound_dir", type=str, default="./data/vggsound/clip"
    )
    args = parser.parse_args()

    vggsound_dir = Path(args.vggsound_dir)
    raw_data_df = pd.read_csv(vggsound_dir / "label.csv", header=None)
    raw_data_df.columns = ["yid", "start", "label", "split"]
    raw_data_df["audio_id"] = raw_data_df.apply(
        lambda row: f'{row["yid"]}_{row["start"]}.mp4', axis=1
    )

    feature_csv_path = Path(args.feature_csv_path)
    feature_df = pd.read_csv(feature_csv_path, sep="\t")

    target_vggsound_dir = Path(args.target_vggsound_dir)

    for split in ["train", "test", "val"]:
        (target_vggsound_dir / split).mkdir(parents=True, exist_ok=True)

        if split == "train":
            audio_ids = raw_data_df[raw_data_df["split"] == "train"
                                   ]["audio_id"].values
        else:
            audio_ids = raw_data_df[raw_data_df["split"] == "test"
                                   ]["audio_id"].values
        audio_ids = set(audio_ids)
        feature_split_df = feature_df[feature_df["audio_id"].isin(audio_ids)]

        with open(target_vggsound_dir / split / "audio.jsonl", "w") as \
            audio_writer, open(target_vggsound_dir / split / "content.jsonl",
            "w") as feature_writer:
            for _, row in tqdm(
                feature_split_df.iterrows(),
                total=len(feature_split_df),
                desc=f'{split}'
            ):
                audio_writer.write(
                    json.dumps({
                        "audio_id": row["audio_id"],
                        "audio": str(vggsound_dir / "video" / row["audio_id"]),
                    }) + "\n"
                )
                feature_writer.write(
                    json.dumps({
                        "audio_id": row["audio_id"],
                        "video": row["hdf5_path"],
                    }) + "\n"
                )


if __name__ == "__main__":
    main()
