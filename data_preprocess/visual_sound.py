from pathlib import Path
import json
import argparse

from tqdm import tqdm

from utils.general import read_jsonl_to_mapping


def main(args):
    vggsound_dir = Path(args.processed_vggsound_dir)
    aid_to_content = {}
    aid_to_audio = {}
    for split in ["train", "val", "test"]:
        aid_to_content.update(
            read_jsonl_to_mapping(
                vggsound_dir / split / "content.jsonl", "audio_id", "video"
            )
        )
        aid_to_audio.update(
            read_jsonl_to_mapping(
                vggsound_dir / split / "audio.jsonl", "audio_id", "audio"
            )
        )

    output_dir = Path(args.target_dir)
    raw_data_dir = Path(args.raw_data_dir)
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

        with open(output_dir / split / "audio.jsonl", "w") as audio_writer, \
            open(output_dir / split / "content.jsonl", "w") as video_writer, \
            open(raw_data_dir / f"visualsound_{split}.txt") as reader:
            for line in reader.readlines():
                yid = line.strip()[:11]
                start = int(line.strip().split("_")[-2]) // 1000
                audio_id = f"{yid}_{start}.mp4"
                if audio_id not in aid_to_audio:
                    print(f"{audio_id} not found in available VGGSound")
                    continue
                audio_writer.write(
                    json.dumps({
                        "audio_id": audio_id,
                        "audio": aid_to_audio[audio_id],
                    }) + "\n"
                )

                video_writer.write(
                    json.dumps({
                        "audio_id": audio_id,
                        "video": aid_to_content[audio_id],
                    }) + "\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument(
        "--processed_vggsound_dir", type=str, default="data/vggsound/clip"
    )
    parser.add_argument(
        "--target_dir", type=str, default="data/visual_sound/clip"
    )
    args = parser.parse_args()
    main(args)
