from pathlib import Path
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audiocaps_csv_dir",
    type=str,
    default="/hpc_stor03/sjtu_home/xuenan.xu/data/audiocaps_v2/"
)
parser.add_argument(
    "--target_audiocaps_dir", type=str, default="data/audiocaps_v2/"
)
parser.add_argument(
    "--waveform_csv",
    type=str,
    help="Path to the AudioSet metadata with hdf5 paths"
)
parser.add_argument(
    "--wav_csv", type=str, help="Path to the full AudioSet metadata"
)
parser.add_argument(
    "--tango_test_ref",
    type=str,
    default="/hpc_stor03/sjtu_home/zeyu.xie/workspace/x2audio/"
    "XToAudioGeneration-master-v1/data_preprocess/audiocaps_tango/"
    "tango_raw_data/test_audiocaps_subset.json"
)
args = parser.parse_args()

AUDIOCAPS_CSV_DIR = Path(args.audiocaps_csv_dir)
TARGET_AUDIOCAPS_DIR = Path(args.target_audiocaps_dir)
TANGO_TEST_REF = args.tango_test_ref

if args.waveform_csv:
    AUDIO_CSV = Path(args.waveform_csv)
elif args.wav_csv:
    AUDIO_CSV = Path(args.wav_csv)
else:
    raise ValueError("Please provide either --waveform_csv or --wav_csv.")


def load_tango_test_ref():
    aid_to_caption = {}
    with open(TANGO_TEST_REF, "r") as f:
        for line in f.readlines():
            item = json.loads(line)
            youtube_id = Path(item["location"]).stem[:11]
            aid_to_caption[youtube_id] = item["captions"]
    return aid_to_caption


def process_audio_id(audio_id: str):
    """Process audio_id to match the Youtube ID format."""
    audio_id = Path(audio_id).stem
    return audio_id[1:12]


audio_df = pd.read_csv(AUDIO_CSV, sep="\t")
audio_df["audio_id"] = audio_df["audio_id"].apply(process_audio_id)
if args.waveform_csv:
    aid_to_fpath = dict(zip(audio_df["audio_id"], audio_df["hdf5_path"]))
elif args.wav_csv:
    aid_to_fpath = dict(zip(audio_df["audio_id"], audio_df["file_name"]))
available_aids = set(audio_df["audio_id"].values)
test_tango_ref = load_tango_test_ref()

for split in ["train", "val", "test"]:
    data_df = pd.read_csv(AUDIOCAPS_CSV_DIR / f"{split}.csv")
    (TARGET_AUDIOCAPS_DIR / split).mkdir(parents=True, exist_ok=True)
    processed_aids = set()
    with open(TARGET_AUDIOCAPS_DIR / split / "audio.jsonl", "w") as audio_writer, \
        open(TARGET_AUDIOCAPS_DIR / split / "caption.jsonl", "w") as text_writer:
        for i, row in data_df.iterrows():
            audio_id = row["youtube_id"]
            if audio_id not in available_aids:
                continue
            if split == "test" and audio_id not in test_tango_ref:
                continue
            if audio_id in processed_aids:
                continue
            audio_writer.write(
                json.dumps({
                    "audio_id": audio_id,
                    "audio": aid_to_fpath[audio_id]
                }) + "\n"
            )
            if split == "test":
                caption = test_tango_ref[audio_id]
            else:
                caption = row["caption"]
            text_writer.write(
                json.dumps({
                    "audio_id": audio_id,
                    "caption": caption
                }) + "\n"
            )
            processed_aids.add(audio_id)
