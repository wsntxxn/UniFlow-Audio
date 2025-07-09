from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm

VGGSOUND_DIR = Path("/cpfs02/shared/speechllm/vggsound/")
FEATURE_CSV_PATH = Path("./data/vggsound/clip_feature.csv")
TARGET_VGGSOUND_DIR = Path("./data/vggsound/clip")


def main():

    raw_data_df = pd.read_csv(VGGSOUND_DIR / "label.csv", header=None)
    raw_data_df.columns = ["yid", "start", "label", "split"]
    raw_data_df["audio_id"] = raw_data_df.apply(
        lambda row: f'{row["yid"]}_{row["start"]}.mp4', axis=1
    )

    feature_df = pd.read_csv(FEATURE_CSV_PATH, sep="\t")

    for split in ["train", "test", "val"]:
        (TARGET_VGGSOUND_DIR / split).mkdir(parents=True, exist_ok=True)

        if split == "train":
            audio_ids = raw_data_df[raw_data_df["split"] == "train"
                                   ]["audio_id"].values
        else:
            audio_ids = raw_data_df[raw_data_df["split"] == "test"
                                   ]["audio_id"].values
        audio_ids = set(audio_ids)
        feature_split_df = feature_df[feature_df["audio_id"].isin(audio_ids)]

        with open(TARGET_VGGSOUND_DIR / split / "audio.jsonl", "w") as \
            audio_writer, open(TARGET_VGGSOUND_DIR / split / "content.jsonl",
            "w") as feature_writer:
            for _, row in tqdm(
                feature_split_df.iterrows(),
                total=len(feature_split_df),
                desc=f'{split}'
            ):
                audio_writer.write(
                    json.dumps({
                        "audio_id": row["audio_id"],
                        "audio": str(VGGSOUND_DIR / "video" / row["audio_id"]),
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
