import os
import subprocess
import argparse
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
    cmd = "/usr/local/bin/ffmpeg"
    assert cmd != "", "activate an env with ffmpeg/ffprobe"

    cmd += " -hide_banner -loglevel error"
    cmd += f" -i {path}"
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{min_side}/min(iw\\,ih):ih*{min_side}/min(iw\\,ih),crop=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd += f" -vcodec {vcodec} -pix_fmt {pix_fmt}"
    cmd += f" -crf {crf}"
    cmd += f" -acodec {acodec} -ar {afps} -ac 1"
    cmd += f" -y {new_path}"
    
    # if not Path(new_path).exists():
    subprocess.call(cmd.split())


def reencode_videos_in_parallel(
    videos, output_dir, fps=21.5, audio_sample_rate=22050, side=256, max_workers=8,
):
    from concurrent.futures import ProcessPoolExecutor
    import concurrent

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                reencode_video,
                video.as_posix(),
                fps,
                audio_sample_rate,
                side,
                output_dir / f"{video.name}",
            ): video
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
        "--origin_video_path",
        "-o",
        type=str,
        required=True,
        help="path to video with different v-fps and sr"
    )
    parser.add_argument(
        "--video_fps",
        "-f",
        type=int,
        required=True,
        help="reencoded vfps"
    )
    parser.add_argument(
        "--audio_fps",
        "-s",
        type=int,
        required=True,
        help="reencoded afps"
    )

    args = parser.parse_args()

    video_dir = Path(args.origin_video_path)
    video_file = sorted(list(video_dir.glob("*.mp4")))

    dir_name = video_dir.name
    out_dir = video_dir.parent / f"{dir_name}-{args.video_fps}fps-{args.audio_fps}hz"

    os.makedirs(str(out_dir), exist_ok=True)

    print(str(out_dir))

    reencode_videos_in_parallel(
        videos=video_file, 
        output_dir=out_dir,
        fps=args.video_fps, 
        audio_sample_rate=args.audio_fps, 
        side=256,
        max_workers=8
    )