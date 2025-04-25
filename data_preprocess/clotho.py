from pathlib import Path
import pandas as pd
import json
import tqdm

PREPARED_AUDIOCAPS_DIR = Path(
    "/cpfs_shared/jiahao.mei/data/tta/clotho/audio"
)
TARGET_AUDIOCAPS_DIR = Path("./data/clotho")

for split in ["dev", "val", "eval"]:
    waveform_df = pd.read_csv(
        PREPARED_AUDIOCAPS_DIR / "annotation" / f"{split}.csv", sep=","
    )
    (TARGET_AUDIOCAPS_DIR / split).mkdir(parents=True, exist_ok=True)
    with open(TARGET_AUDIOCAPS_DIR / split / "audio.jsonl", "w") as audio_writer, \
        open(TARGET_AUDIOCAPS_DIR / split / "caption.jsonl", "w") as caption_writer:
        for _, row in tqdm.tqdm(waveform_df.iterrows()):
            tmp_path = f"/hpc_stor03/sjtu_home/zeyu.xie/maa/x2audio_data/clotho/audio/{split}/{row['file_name']}"
            if Path(tmp_path).exists():
                audio_writer.write(
                    json.dumps({
                        "audio_id": row["file_name"],
                        "audio": f"clotho/audio/{split}/{row['file_name']}"
                    }) + "\n"
                )
            else:
                print(tmp_path)

            for caption_idx in range(1, 6): 
                caption_writer.write(
                    json.dumps({
                        "audio_id": row["file_name"],
                        "caption": row[f"caption_{caption_idx}"]
                    }) + "\n"
                )
