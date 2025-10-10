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
    从视频文件中提取指定时间段的音频，并以指定采样率保存
    
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param start_time: 开始时间(秒或HH:MM:SS格式)
    :param end_time: 结束时间(秒或HH:MM:SS格式)
    :param sample_rate: 音频采样率(默认44100Hz)
    """
    video_path, audio_path = entry

    # 构建ffmpeg命令
    cmd = [
        which_ffmpeg(),
        '-y',  # 覆盖已存在文件
        '-i',
        str(video_path),
        '-ss',
        str(start_time),  # 开始时间
        '-to',
        str(end_time),  # 结束时间
        '-ar',
        str(sample_rate),  # 采样率
        '-ac',
        '1',  # 立体声
        '-q:a',
        '0',  # 最高质量
        str(audio_path)
    ]

    try:
        subprocess.run(cmd, check=True)
        sys.stdout.flush()
    except subprocess.CalledProcessError as e:
        print(f"提取音频失败: {e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"发生错误: {str(e)}")


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
        print(f"输入目录不存在: {audio_jsonl}")
        return

    output_dir_name = output_dir.name
    output_dir = output_dir.with_name(
        f"{output_dir_name}_{sample_rate}Hz_{start_time}s_to_{end_time}s"
    )

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
    output_jsonl_name = output_jsonl.stem
    output_jsonl = output_jsonl.with_stem(
        f"{output_jsonl_name}_{sample_rate}Hz_{start_time}s_to_{end_time}s"
    )
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

    print("所有文件处理完成")
    sys.stdout.flush()
    os.system("stty sane")  # 修复终端状态


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
