import json
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
import concurrent
from pathlib import Path

from tqdm import tqdm

VCODEC = "h264"
CRF = 10
PIX_FMT = "yuv420p"
ACODEC = "aac"


def reencode_video(
    path,
    vfps,
    afps,
    min_side,
    new_path,
    acodec=ACODEC,
    vcodec=VCODEC,
    pix_fmt=PIX_FMT,
    crf=CRF,
):
    # reencode the original mp4: rescale, resample video and resample audio
    cmd = f"ffmpeg -hide_banner -loglevel error -i {path} -vf " \
          f"fps={vfps},scale=iw*{min_side}/min(iw\\,ih):ih*{min_side}/min(iw\\,ih),crop=trunc(iw/2)*2:trunc(ih/2)*2 " \
          f"-vcodec {vcodec} -pix_fmt {pix_fmt} -crf {crf} -acodec {acodec} -ar {afps} -ac 1 " \
          f"-y {new_path}"

    # if not Path(new_path).exists():
    subprocess.call(cmd.split())


def reencode_videos_in_parallel(
    videos,
    output_dir,
    fps=21.5,
    audio_sample_rate=22050,
    side=256,
    max_workers=8,
):

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                reencode_video,
                video.__str__(),
                fps,
                audio_sample_rate,
                side,
                output_dir / f"{Path(video).name}",
            ):
                video
            for video in videos
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reencoding videos",
        ):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        "-i",
        type=str,
        required=True,
        help="path to video with different v-fps and sr"
    )
    parser.add_argument(
        "--video_fps", "-f", type=int, required=True, help="reencoded vfps"
    )
    parser.add_argument(
        "--audio_fps", "-s", type=int, required=True, help="reencoded afps"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "--output_jsonl", "-j", type=str, required=True, help="output jsonl file"
    )

    args = parser.parse_args()

    if Path(args.videos).is_dir():
        videos = sorted(Path(args.videos).glob("*.mp4"))
    elif args.videos.endswith(".txt"):
        with open(args.videos, "r") as f:
            videos = [line.strip() for line in f.readlines()]
    elif args.videos.endswith(".jsonl"):
        with open(args.videos, "r") as f:
            videos = [json.loads(line)["video"] for line in f.readlines()]

    out_dir = Path(args.output_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    print(str(out_dir))

    reencode_videos_in_parallel(
        videos=videos,
        output_dir=out_dir,
        fps=args.video_fps,
        audio_sample_rate=args.audio_fps,
        side=256,
        max_workers=8
    )

    with open(args.output_jsonl, "w") as f:
        for file in out_dir.glob("*.mp4"):
            f.write(json.dumps({"audio_id": file.name, "video": str(file)}) + "\n")
