import json
from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--musicaps_csv", type=str, help="Path to the MusicCaps CSV file"
)
parser.add_argument(
    "--wav_csv", type=str, help="Path to the full AudioSet metadata"
)
parser.add_argument(
    "--waveform_csv",
    type=str,
    help="Path to the AudioSet metadata with hdf5 paths"
)
parser.add_argument(
    "--target_musicaps_dir", type=str, default="data/music_caps/"
)

args = parser.parse_args()

MUSICAPS_CSV = Path(args.musicaps_csv)
TARGET_MUSICAPS_DIR = Path(args.target_musicaps_dir)

if args.waveform_csv:
    AUDIO_CSV = Path(args.waveform_csv)
elif args.wav_csv:
    AUDIO_CSV = Path(args.wav_csv)
else:
    raise ValueError("Please provide either --waveform_csv or --wav_csv.")

audio_df = pd.read_csv(AUDIO_CSV, sep="\t")
audio_df["audio_id"] = audio_df["audio_id"].apply(lambda x: x[1:12])
if args.waveform_csv:
    aid_to_fpath = dict(zip(audio_df["audio_id"], audio_df["hdf5_path"]))
elif args.wav_csv:
    aid_to_fpath = dict(zip(audio_df["audio_id"], audio_df["file_name"]))
available_aids = set(audio_df["audio_id"].values)

data_df = pd.read_csv(MUSICAPS_CSV)
TARGET_MUSICAPS_DIR.mkdir(parents=True, exist_ok=True)
with open(TARGET_MUSICAPS_DIR / "audio.jsonl", "w") as audio_writer, \
    open(TARGET_MUSICAPS_DIR / "caption.jsonl", "w") as text_writer:
    for i, row in data_df.iterrows():
        audio_id = row["ytid"]
        if audio_id not in available_aids:
            continue
        audio_writer.write(
            json.dumps({
                "audio_id": audio_id,
                "audio": aid_to_fpath[audio_id]
            }) + "\n"
        )

        caption = row["caption"]
        text_writer.write(
            json.dumps({
                "audio_id": audio_id,
                "caption": caption
            }) + "\n"
        )
