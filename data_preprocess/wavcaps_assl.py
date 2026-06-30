from pathlib import Path
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wavcaps_assl_json_path",
    type=str,
    default=
    "/mnt/cloudstorfs/public/shared/data/raa/WavCaps/json_orig/as_final.json"
)
parser.add_argument("--target_dir", type=str, default="./data/wavcaps_assl")
parser.add_argument(
    "--audiocaps_test_csv_path",
    type=str,
    default="/mnt/cloudstorfs/sjtu_home/xuenan.xu/data/audiocaps_v2/test.csv"
)
parser.add_argument(
    "--waveform_csv",
    type=str,
    help="Path to the AudioSet metadata with hdf5 paths"
)
parser.add_argument(
    "--wav_csv", type=str, help="Path to the full AudioSet metadata"
)
args = parser.parse_args()

WAVCAPS_ASSL_JSON_PATH = Path(args.wavcaps_assl_json_path)
TARGET_DIR = Path(args.target_dir)
AUDIOCAPS_TEST_CSV_PATH = Path(args.audiocaps_test_csv_path)
if args.waveform_csv:
    AUDIO_CSV = Path(args.waveform_csv)
elif args.wav_csv:
    AUDIO_CSV = Path(args.wav_csv)
else:
    raise ValueError("Please provide either --waveform_csv or --wav_csv.")

audio_df = pd.read_csv(AUDIO_CSV, sep="\t")
if args.waveform_csv:
    aid_to_fpath = dict(zip(audio_df["audio_id"], audio_df["hdf5_path"]))
elif args.wav_csv:
    audio_df["audio_id"] = audio_df["audio_id"].apply(
        lambda x: Path(x).stem[1:12]
    )
    aid_to_fpath = dict(zip(audio_df["audio_id"], audio_df["file_name"]))
available_aids = set(audio_df["audio_id"].values)

audiocaps_test_df = pd.read_csv(AUDIOCAPS_TEST_CSV_PATH)
audiocaps_test_yids = set(audiocaps_test_df["youtube_id"].values)

TARGET_DIR.mkdir(exist_ok=True, parents=True)

with open(WAVCAPS_ASSL_JSON_PATH, "r") as f:
    data = json.load(f)

with open(TARGET_DIR / "audio.jsonl", "w") as audio_writer, \
    open(TARGET_DIR / "caption.jsonl", "w") as text_writer:
    for item in data["data"]:
        audio_id = Path(item["id"]).stem[1:12]
        if audio_id not in available_aids:
            continue
        if audio_id in audiocaps_test_yids:
            continue

        audio_writer.write(
            json.dumps({
                "audio_id": audio_id,
                "audio": aid_to_fpath[audio_id]
            }) + "\n"
        )
        text_writer.write(
            json.dumps({
                "audio_id": audio_id,
                "caption": item["caption"]
            }) + "\n"
        )
