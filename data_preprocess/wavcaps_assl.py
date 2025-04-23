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
    default="/mnt/cloudstorfs/public/shared/data/raa/AudioSet/unbalanced_train/"
    "waveform/waveform.csv"
)
args = parser.parse_args()

WAVCAPS_ASSL_JSON_PATH = Path(args.wavcaps_assl_json_path)
TARGET_DIR = Path(args.target_dir)
AUDIOCAPS_TEST_CSV_PATH = Path(args.audiocaps_test_csv_path)
WAVEFORM_CSV = Path(args.waveform_csv)

waveform_df = pd.read_csv(WAVEFORM_CSV, sep="\t")
aid_to_h5path = dict(zip(waveform_df["audio_id"], waveform_df["hdf5_path"]))
available_aids = set(waveform_df["audio_id"].values)

audiocaps_test_df = pd.read_csv(AUDIOCAPS_TEST_CSV_PATH)
audiocaps_test_yids = set(audiocaps_test_df["youtube_id"].values)

TARGET_DIR.mkdir(exist_ok=True, parents=True)

with open(WAVCAPS_ASSL_JSON_PATH, "r") as f:
    data = json.load(f)

with open(TARGET_DIR / "audio.jsonl", "w") as audio_writer, \
    open(TARGET_DIR / "text.jsonl", "w") as text_writer:
    for item in data["data"]:
        audio_id = item["id"]
        if audio_id not in available_aids:
            continue
        youtube_id = audio_id[1:-4]
        if youtube_id in audiocaps_test_yids:
            continue
        h5path = aid_to_h5path[audio_id]
        audio_writer.write(
            json.dumps({
                "audio_id": audio_id,
                "audio": h5path
            }) + "\n"
        )
        text_writer.write(
            json.dumps({
                "audio_id": audio_id,
                "text": item["caption"]
            }) + "\n"
        )
