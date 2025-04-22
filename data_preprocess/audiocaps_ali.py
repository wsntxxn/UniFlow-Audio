from pathlib import Path
import pandas as pd
import json
import tqdm

PREPARED_AUDIOCAPS_DIR = Path(
    "/hpc_stor03/sjtu_home/xuenan.xu/workspace/audio_captioning/data/audiocaps"
)
TARGET_AUDIOCAPS_DIR = Path("/hpc_stor03/sjtu_home/zeyu.xie/maa/audiocaps/jsonl")

for split in ["train", "val", "test"]:
    waveform_df = pd.read_csv(
        PREPARED_AUDIOCAPS_DIR / split / "waveform.csv", sep="\t"
    )
    (TARGET_AUDIOCAPS_DIR / split).mkdir(parents=True, exist_ok=True)
    with open(TARGET_AUDIOCAPS_DIR / split / "audio.jsonl", "w") as writer:
        for _, row in tqdm.tqdm(waveform_df.iterrows()):
            tmp_path = f"/hpc_stor03/sjtu_home/zeyu.xie/maa/audiocaps/audio/{split}/{row['audio_id']}"
            if Path(tmp_path).exists():
                writer.write(
                    json.dumps({
                        "audio_id": row["audio_id"],
                        "audio": f"audiocaps/audio/{split}/{row['audio_id']}"
                    }) + "\n"
                )
            else:
                print(tmp_path)
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
