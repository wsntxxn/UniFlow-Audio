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
    default="/mnt/cloudstorfs/public/shared/data/raa/AudioSet/unbalanced_train/"
    "waveform/waveform.csv"
)
parser.add_argument(
    "--tango_test_ref",
    type=str,
    default="/hpc_stor03/sjtu_home/zeyu.xie/workspace/x2audio/"
    "XToAudioGeneration-master/data_preprocess/audiocaps_tango/"
    "tango_raw_data/test_audiocaps_subset.json"
)
args = parser.parse_args()

AUDIOCAPS_CSV_DIR = Path(args.audiocaps_csv_dir)
TARGET_AUDIOCAPS_DIR = Path(args.target_audiocaps_dir)
WAVEFORM_CSV = Path(args.waveform_csv)
TANGO_TEST_REF = args.tango_test_ref


def load_tango_test_ref():
    aid_to_caption = {}
    with open(TANGO_TEST_REF, "r") as f:
        for line in f.readlines():
            item = json.loads(line)
            youtube_id = Path(item["location"]).stem[:11]
            audio_id = f"Y{youtube_id}.wav"
            aid_to_caption[audio_id] = item["captions"]
    return aid_to_caption


waveform_df = pd.read_csv(WAVEFORM_CSV, sep="\t")
aid_to_h5path = dict(zip(waveform_df["audio_id"], waveform_df["hdf5_path"]))
available_aids = set(waveform_df["audio_id"].values)
test_tango_ref = load_tango_test_ref()

for split in ["train", "val", "test"]:
    data_df = pd.read_csv(AUDIOCAPS_CSV_DIR / f"{split}.csv")
    (TARGET_AUDIOCAPS_DIR / split).mkdir(parents=True, exist_ok=True)
    processed_aids = set()
    with open(TARGET_AUDIOCAPS_DIR / split / "audio.jsonl", "w") as audio_writer, \
        open(TARGET_AUDIOCAPS_DIR / split / "text.jsonl", "w") as text_writer:
        for i, row in data_df.iterrows():
            youtube_id = row["youtube_id"]
            audio_id = f"Y{youtube_id}.wav"
            if audio_id not in available_aids:
                continue
            if split == "test" and audio_id not in test_tango_ref:
                continue
            if audio_id in processed_aids:
                continue
            audio_writer.write(
                json.dumps(
                    {
                        "audio_id": audio_id,
                        "audio": aid_to_h5path[audio_id]
                    }
                ) + "\n"
            )
            if split == "test":
                caption = test_tango_ref[audio_id]
            else:
                caption = row["caption"]
            text_writer.write(
                json.dumps({
                    "audio_id": audio_id,
                    "text": caption
                }) + "\n"
            )
            processed_aids.add(audio_id)
