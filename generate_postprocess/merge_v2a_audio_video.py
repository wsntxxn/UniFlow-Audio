import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.video import merge_audio_video
from utils.general import read_jsonl_to_mapping


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.aid_video_mapping.endswith('.csv'):
        df = pd.read_csv(args.aid_video_mapping, sep='\t')
        aid_to_video = dict(zip(df['audio_id'], df['video_path']))
    elif args.aid_video_mapping.endswith('.jsonl'):
        aid_to_video = read_jsonl_to_mapping(
            args.aid_video_mapping, "audio_id", "audio"
        )
        aid_to_video = {Path(k).stem: v for k, v in aid_to_video.items()}
    files = list(Path(args.audio_path).glob('*.wav'))

    if args.num_samples is not None:
        files = files[:args.num_samples]

    for audio_file in tqdm(files):
        audio_id = audio_file.stem
        if audio_id.endswith('_dummy'):
            audio_id = audio_id[:-6]
        if audio_id not in aid_to_video:
            continue
        video_file = aid_to_video[audio_id]
        output_file = output_dir / f"{audio_id}.mp4"
        merge_audio_video(audio_file, video_file, output_file, args.backend)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aid_video_mapping",
        type=str,
        required=True,
        help="mapping file between audio id and video files"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="path to the audio directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "--backend",
        choices=["moviepy", "ffmpeg"],
        default="ffmpeg",
        help="backend for merging audio and video"
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=None,
        help="number of samples to process, default is all"
    )
    args = parser.parse_args()
    main(args)
