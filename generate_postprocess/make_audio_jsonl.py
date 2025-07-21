# python generate_postprocess/make_audio_jsonl.py  \
#     --audio_dir "/cpfs_shared/jiahao.mei/code/x_to_audio_generation/experiments/DummyContentAudioDiffusion/double_content+cross_attn_adapter_init_8gpus/inference/text_to_audio" \
#     --task tta \
#     --output_file "./tta_files.jsonl"

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from utils.general import transform_gen_fn_to_id


def generate_jsonl(args) -> None:
    audio_dir = Path(args.audio_dir)
    task = args.task
    audio_files = sorted(audio_dir.iterdir())

    Path(args.output_file).parent.mkdir(exist_ok=True, parents=True)

    with open(args.output_file, 'w') as writer:
        for audio_file in tqdm(audio_files):
            audio_id = transform_gen_fn_to_id(audio_file, task)

            if audio_file.suffix != '.wav':
                # "should hold only wav in dir"
                continue

            writer.write(
                json.dumps(
                    {
                        "audio_id": audio_id,
                        "audio": str(audio_file.resolve())
                    },
                    ensure_ascii=False,
                ) + "\n"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        "-d",
        type=str,
        required=True,
        help="directory of audio files"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["tts", "tta", "ttm", "se", "sr", "svs", "v2a"],
        required=True,
        help="task name"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="output file name"
    )

    args = parser.parse_args()
    generate_jsonl(args)
