import argparse
from pathlib import Path

import pandas as pd
from accel_hydra.utils.general import read_jsonl_to_mapping
from tqdm import tqdm

from utils.video import merge_audio_video


def main(args):
    if args.output_dir is None:
        audio_path_name = Path(args.audio_path).name
        output_dir = Path(args.audio_path
                         ).with_name(f"{audio_path_name}_video")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.aid_video_mapping.endswith('.csv'):
        df = pd.read_csv(args.aid_video_mapping, sep='\t')
        df["audio_id"] = df["audio_id"].apply(lambda x: "_".join(x.split()))
        aid_to_video = dict(zip(df['audio_id'], df['video_path']))
    elif args.aid_video_mapping.endswith('.jsonl'):
        aid_to_video = read_jsonl_to_mapping(
            args.aid_video_mapping, "audio_id", "audio"
        )
        aid_to_video = {Path(k).stem: v for k, v in aid_to_video.items()}

    yid_to_video = {
        Path(video_path).stem: video_path
        for aid, video_path in aid_to_video.items()
    }
    files = list(Path(args.audio_path).glob('*.wav'))

    if args.num_samples is not None:
        files = files[:args.num_samples]

    for audio_file in tqdm(files):
        audio_id = audio_file.stem
        if audio_id.endswith('_dummy'):
            audio_id = audio_id[:-6]
        if audio_id in aid_to_video:
            video_file = aid_to_video[audio_id]
        elif audio_id in yid_to_video:
            video_file = yid_to_video[audio_id]
        else:
            print(f"{audio_id} not found in videos")
            continue
        output_file = output_dir / f"{audio_id}.mp4"
        merge_audio_video(
            audio_file,
            video_file,
            output_file,
            args.backend,
            low_quality=args.low_quality
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aid_video_mapping",
        "-m",
        type=str,
        required=True,
        help="mapping file between audio id and video files"
    )
    parser.add_argument(
        "--audio_path",
        "-a",
        type=str,
        required=True,
        help="path to the audio directory"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=False,
        help="output directory"
    )
    parser.add_argument(
        "--backend",
        "-b",
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
    parser.add_argument(
        "--low_quality",
        action="store_true",
        help=
        "use lower quality settings to reduce file size (higher CRF, lower resolution for ffmpeg backend)"
    )
    args = parser.parse_args()
    main(args)
