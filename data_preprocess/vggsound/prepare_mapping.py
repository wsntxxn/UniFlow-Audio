from pathlib import Path
import pandas as pd

VGGSOUND_DIR = Path("/cpfs02/shared/speechllm/vggsound/")
TARGET_FILE = "./data/vggsound/mapping.csv"

df = pd.read_csv(VGGSOUND_DIR / "label.csv", header=None)
df.columns = ["yid", "start", "label", "split"]
df["audio_id"] = df.apply(
    lambda x: f"{x['yid']}_{x['start']}_{x['label']}", axis=1
)
df["video_path"] = df.apply(
    lambda x: str(VGGSOUND_DIR / "video" / f"{x['yid']}_{x['start']}.mp4"),
    axis=1
)

df[["audio_id", "video_path"]].to_csv(TARGET_FILE, index=False, sep="\t")
