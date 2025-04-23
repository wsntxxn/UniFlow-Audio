from pathlib import Path
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prepared_audiocaps_dir",
    type=str,
    default=
    "/hpc_stor03/sjtu_home/xuenan.xu/workspace/audio_captioning/data/audiocaps"
)
parser.add_argument(
    "--target_audiocaps_dir", type=str, default="data/audiocaps_v2/"
)
parser.add_argument(
    "--tango_test_ref",
    type=str,
    default="/hpc_stor03/sjtu_home/zeyu.xie/workspace/x2audio/"
    "XToAudioGeneration-master/data_preprocess/audiocaps_tango/"
    "tango_raw_data/test_audiocaps_subset.json"
)
args = parser.parse_args()
PREPARED_AUDIOCAPS_DIR = Path(args.prepared_audiocaps_dir)
TARGET_AUDIOCAPS_DIR = Path(args.target_audiocaps_dir)
TANGO_TEST_REF = Path(args.tango_test_ref)


def load_tango_test_ref():
    aid_to_caption = {}
    with open(TANGO_TEST_REF, "r") as f:
        for line in f.readlines():
            item = json.loads(line)
            youtube_id = Path(item["location"]).stem[:11]
            audio_id = f"Y{youtube_id}.wav"
            aid_to_caption[audio_id] = item["captions"]
    return aid_to_caption


test_tango_ref = load_tango_test_ref()

for split in ["train", "val", "test"]:
    waveform_df = pd.read_csv(
        PREPARED_AUDIOCAPS_DIR / split / "waveform.csv", sep="\t"
    )
    (TARGET_AUDIOCAPS_DIR / split).mkdir(parents=True, exist_ok=True)
    with open(TARGET_AUDIOCAPS_DIR / split / "audio.jsonl", "w") as writer:
        for _, row in waveform_df.iterrows():
            audio_id = row["audio_id"]
            if split == "test" and audio_id not in test_tango_ref:
                continue
            writer.write(
                json.dumps({
                    "audio_id": audio_id,
                    "audio": row["hdf5_path"]
                }) + "\n"
            )

    caption_data = json.load(
        open(PREPARED_AUDIOCAPS_DIR / split / "text.json")
    )
    with open(TARGET_AUDIOCAPS_DIR / split / "caption.jsonl", "w") as writer:
        for audio_item in caption_data["audios"]:
            audio_id = audio_item["audio_id"]
            if split == "test" and audio_id not in test_tango_ref:
                continue

            if split == "test":
                caption = test_tango_ref[audio_id]
            else:
                caption = audio_item["captions"][0]["caption"]

            writer.write(
                json.dumps({
                    "audio_id": audio_id,
                    "caption": caption
                }) + "\n"
            )
