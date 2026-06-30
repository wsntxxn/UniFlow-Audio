import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm

EXCLUDE_PATTERNS = (
    r"^\s*none\s*$",
    r"^\s*no\b.*",
    r".*\babsent\b.*",
    r".*\bconfidence\b.*",
)


def is_invalid_text(text: str) -> bool:
    value = text.strip().lower()
    if not value:
        return True
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, value):
            return True
    return False


def main(args):
    acavcaps_root = Path(args.acavcaps_root)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "audio.jsonl", "w") as audio_writer, open(
        output_dir / "caption.jsonl", "w"
    ) as caption_writer:
        pbar = tqdm(
            acavcaps_root.rglob("*.flac"),
            unit="file",
        )

        for audio_file in pbar:
            caption_file = audio_file.with_suffix(".json")

            with open(caption_file, "r", encoding="utf-8") as f:
                caption_obj = json.load(f)

            audio_id = audio_file.stem
            if args.data_field == "music":
                captions = caption_obj["music"]
            elif args.data_field == "long":
                captions = caption_obj["long"]
            elif args.data_field == "short":
                captions = caption_obj["short"]
            else:
                captions = caption_obj["long"] + caption_obj["short"]

            if args.caption_mode == "random":
                captions = [random.choice(captions)]

            for caption_idx, caption in enumerate(captions):
                data_id = f"{audio_id}_{caption_idx + 1}"

                if is_invalid_text(caption):
                    continue

                audio_writer.write(
                    json.dumps({
                        "audio_id":
                            data_id,
                        "audio":
                            audio_file.as_posix().
                            replace("/mnt/brainllm_s3/data/", "s3://data/"),
                        "duration":
                            10.0,
                    }) + "\n"
                )
                caption_writer.write(
                    json.dumps({
                        "audio_id": data_id,
                        "caption": caption,
                    }) + "\n"
                )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acavcaps_root",
        "-r",
        type=str,
        default="/mnt/brainllm_s3/data/ACAVCAPS/extracted/0M0"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="data/acavcaps_0m0"
    )
    parser.add_argument(
        "--data_field",
        "-f",
        type=str,
        default="music",
        choices=["music", "overall", "long", "short"],
        help="The field in the JSON file that contains the caption text."
    )
    parser.add_argument(
        "--caption_mode",
        "-m",
        type=str,
        default="all",
        choices=["all", "random"],
        help=
        "Whether to use all captions or randomly select one caption per audio."
    )
    args = parser.parse_args()

    main(args)
