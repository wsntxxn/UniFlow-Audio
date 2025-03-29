from pathlib import Path
import pandas as pd
import json
import shutil
from tqdm import tqdm

# PREPARED_VGGSOUND_DIR = Path("/hpc_stor03/public/shared/data/raa/VGGSound/")
TARGET_VGGSOUND_DIR = Path("./data/vggsound-cavp")

def main():
    waveform_df = pd.read_csv(
        "/hpc_stor03/sjtu_home/yaoyun.zhang/work/XToAudioGeneration/data/vggsound/vggsound_cavp_metadata.csv"
    )

    for split in ["train", "test", "val"]:
        (TARGET_VGGSOUND_DIR / split).mkdir(parents=True, exist_ok=True)

        with open(TARGET_VGGSOUND_DIR / split / "audio.jsonl", "w") as audio_writer:
            for _, row in tqdm(waveform_df.iterrows(), total=len(waveform_df), desc=f'{split}'):
                vid_id = f'{row["name"]}_{row["target"]}.mp4'
                vid_path = row["h5_path"]
                if row["split"] == split:
                    audio_writer.write(
                        json.dumps({
                            "audio_id": vid_id,
                            "audio": str(vid_path),
                        }) + "\n"
                    )
                
        with open(TARGET_VGGSOUND_DIR / split / "content.jsonl", "w") as caption_writer:
            for _, row in tqdm(waveform_df.iterrows(), total=len(waveform_df), desc=f'{split}'):
                vid_id = f'{row["name"]}_{row["target"]}.mp4'
                vid_path = row["h5_path"]
                if row["split"] == split:
                    caption_writer.write(
                        json.dumps({
                            "audio_id": vid_id,
                            "video": str(vid_path),
                        }) + "\n"
                    )

        if split == "val":
            test_jsonl_audio = TARGET_VGGSOUND_DIR / "test" / "audio.jsonl"
            test_jsonl_content = TARGET_VGGSOUND_DIR / "test" / "content.jsonl"
            shutil.copy(test_jsonl_audio, TARGET_VGGSOUND_DIR / "val" / "audio.jsonl")
            shutil.copy(test_jsonl_content, TARGET_VGGSOUND_DIR / "val" / "content.jsonl")
            break

if __name__ == "__main__":
    main()