from pathlib import Path
import pandas as pd
import json

PREPARED_AUDIOCAPS_DIR = Path(
    "/hpc_stor03/sjtu_home/xuenan.xu/workspace/audio_captioning/data/audiocaps"
)
TARGET_AUDIOCAPS_DIR = Path("./data/audiocaps")

for split in ["train", "val", "test"]:
    waveform_df = pd.read_csv(
        PREPARED_AUDIOCAPS_DIR / split / "waveform.csv", sep="\t"
    )
    (TARGET_AUDIOCAPS_DIR / split).mkdir(parents=True, exist_ok=True)
    with open(TARGET_AUDIOCAPS_DIR / split / "audio.jsonl", "w") as writer:
        for _, row in waveform_df.iterrows():
            writer.write(
                json.dumps({
                    "audio_id": row["audio_id"],
                    "audio": row["hdf5_path"]
                }) + "\n"
            )

    caption_data = json.load(
        open(PREPARED_AUDIOCAPS_DIR / split / "text.json")
    )
    with open(TARGET_AUDIOCAPS_DIR / split / "caption.jsonl", "w") as writer:
        for audio_item in caption_data["audios"]:
            for caption_item in audio_item["captions"]:
                writer.write(
                    json.dumps({
                        "audio_id": audio_item["audio_id"],
                        "caption": caption_item["caption"]
                    }) + "\n"
                )
