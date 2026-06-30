import argparse
import os
import json
import tempfile
from io import BytesIO
from multiprocessing import Pool

import librosa
import soundfile as sf
from tqdm import tqdm
from petrel_client.client import Client

PETREL_OSS_CONFIG = "/mnt/shared-storage-user/xuxuenan/petreloss.conf"
PETREL_CLIENT = Client(PETREL_OSS_CONFIG)


def resolve_audio_path(audio_path: str, base_path: str | None = None) -> str:
    """
    If `audio_path` is relative and `base_path` is provided, join them.
    If `audio_path` is already absolute, return it.
    """
    if base_path and not os.path.isabs(audio_path):
        return os.path.join(base_path, audio_path)
    return audio_path


def add_duration(item: dict) -> dict:

    path = item["path"]
    if path.startswith("s3://"):
        if not PETREL_CLIENT.contains(path):
            return {}
        bytes_data = PETREL_CLIENT.get(path)
        if path.endswith(".mp4"):
            fd, path = tempfile.mkstemp(suffix=".mp4")
            with open(path, "wb") as f:
                f.write(bytes_data)
            duration = librosa.core.get_duration(path=path)
            os.close(fd)
            os.remove(path)
        else:
            info = sf.info(BytesIO(bytes_data))
            duration = info.duration
    else:
        if not os.path.exists(item["path"]):
            return {}
        duration = librosa.core.get_duration(path=path)
    return {
        "audio_id": item["audio_id"],
        "audio": item["audio"],
        "duration": duration
    }


def process_jsonl(
    input_jsonl: str,
    workers: int,
    base_path: str | None = None,
):
    items = []
    with open(input_jsonl, "r") as f:
        for line in f:
            item = json.loads(line)
            audio_path = item["audio"]
            audio_path = resolve_audio_path(audio_path, base_path)
            items.append({
                "audio_id": item["audio_id"],
                "path": audio_path,
                "audio": item["audio"]
            })

    with Pool(processes=workers) as pool:
        result = list(tqdm(pool.imap(add_duration, items), total=len(items)))

    if all([len(x) == 0 for x in result]):
        raise Exception("No valid duration found")
    else:
        with open(input_jsonl, "w") as f:
            for item in result:
                if len(item) > 0:
                    f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Add duration (seconds) into each line of audio.jsonl."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input jsonl file or a directory containing audio.jsonl files.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="If audio path in jsonl is relative, prefix it with base_path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread workers for duration extraction (uses torchaudio).",
    )

    args = parser.parse_args()

    process_jsonl(
        input_jsonl=args.input,
        base_path=args.base_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
