import argparse
import json

import pandas as pd


def main(args):
    df = pd.read_csv("./data/libritts/duration.csv", sep="\t")
    aid_to_duration = dict(zip(df["audio_id"], df["duration"]))

    for split in ["train", "val", "test"]:
        orig_file = f"./data/libritts/{split}/phoneme.jsonl"
        filtered_file = f"./data/libritts/{split}/phoneme_duration_filtered.jsonl"
        with open(orig_file, "r") as f_in, open(filtered_file, "w") as f_out:
            for line in f_in:
                item = json.loads(line.strip())
                audio_id = item["audio_id"]
                duration = aid_to_duration[audio_id.lstrip("0") + ".wav"]
                if duration <= args.max_threshold and duration >= args.min_threshold:
                    f_out.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_threshold',
        type=float,
        default=20.0,
        help='maximum duration threshold'
    )
    parser.add_argument(
        '--min_threshold',
        type=float,
        default=0.0,
        help='minimum duration threshold'
    )

    args = parser.parse_args()

    main(args)
