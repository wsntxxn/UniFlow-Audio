import argparse
from pathlib import Path
import sys
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
from tqdm import tqdm

from utils.general import sanitize_filename


def which_ffmpeg() -> str:
    """Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    """
    result = subprocess.run(["which", "ffmpeg"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode("utf-8").replace("\n", "")
    return ffmpeg_path


def extract_audio_segment(
    entry, output_dir, start_time, end_time, sample_rate=44100
):
    """
    Extract audio segment from video file with specified time range and sample rate
    
    :param entry: Tuple containing video_path and audio_path
    :param output_dir: Output directory
    :param start_time: Start time (seconds or HH:MM:SS format)
    :param end_time: End time (seconds or HH:MM:SS format)
    :param sample_rate: Audio sample rate (default 44100Hz)
    """
    video_path, audio_path = entry

    ffmpeg_bin = which_ffmpeg()

    if ffmpeg_bin == "":
        raise ValueError("ffmpeg not found")

    # Build ffmpeg command
    cmd = [
        which_ffmpeg(),
        '-y',  # Overwrite existing files
        '-i',
        str(video_path),
        '-ss',
        str(start_time),  # Start time
        '-to',
        str(end_time),  # End time
        '-ar',
        str(sample_rate),  # Sample rate
        '-ac',
        '1',  # Mono (single channel)
        '-q:a',
        '0',  # Highest quality
        str(audio_path)
    ]

    try:
        subprocess.run(cmd, check=True)
        sys.stdout.flush()
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio: {e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def process_directory(
    audio_jsonl,
    output_dir,
    output_jsonl,
    label_path,
    start_time,
    end_time,
    sample_rate=44100
):
    """
    """
    audio_jsonl = Path(audio_jsonl)
    output_dir = Path(output_dir)

    if not audio_jsonl.exists():
        print(f"Input directory does not exist: {audio_jsonl}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(label_path, header=None)
    mapping = {}
    for _, row in label_df.iterrows():
        mapping[f"{row[0]}_{row[1]}"] = row[2]

    worker = partial(
        extract_audio_segment,
        output_dir=output_dir,
        start_time=start_time,
        end_time=end_time,
        sample_rate=sample_rate,
    )

    output_jsonl = Path(output_jsonl)
    with open(audio_jsonl, "r") as reader, open(output_jsonl, "w") as writer:
        lines = reader.readlines()
        entries = []
        for line in lines:
            item = json.loads(line.strip())
            video_file = item["video"]
            video_fname = Path(video_file).stem
            audio_file = output_dir / sanitize_filename(
                f"{video_fname}_{mapping[video_fname]}.wav"
            )
            entries.append((video_file, audio_file))
            writer.write(
                json.dumps({
                    "audio_id": item["audio_id"],
                    "audio": audio_file.resolve().__str__()
                }) + "\n"
            )

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(worker, entries), total=len(entries)))

    print("All files processed")
    sys.stdout.flush()
    os.system("stty sane")  # Restore terminal state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_jsonl",
        help="JSONL file to indicate audio files to process",
        default="data/vggsound/clip/test/audio.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output audio directory",
        default="data/vggsound/test_wav",
    )
    parser.add_argument(
        "--output_jsonl",
        help="JSONL file to save processed audio files",
        default="data/vggsound/test_audio.jsonl"
    )
    parser.add_argument(
        "--start_time", help="start time(s or HH:MM:SS format)", default=0
    )
    parser.add_argument(
        "--end_time", help="end time(s or HH:MM:SS format)", default=10.0
    )
    parser.add_argument(
        "--vggsound_label_path",
        type=str,
        default="/cpfs02/shared/speechllm/VGGSound/label.csv"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="default sample rate required by LDM-Eval"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers"
    )

    args = parser.parse_args()

    process_directory(
        args.audio_jsonl, args.output_dir, args.output_jsonl,
        args.vggsound_label_path, args.start_time, args.end_time,
        args.sample_rate
    )
