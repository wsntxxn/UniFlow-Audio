from pathlib import Path
import os
from moviepy import VideoFileClip, AudioFileClip
from moviepy.audio.fx import AudioLoop


def merge_audio_video(
    audio_path: str | Path,
    video_path: str | Path,
    target_path: str | Path,
    backend: str = "moviepy",
    logging: bool = False
):
    """
    Merge audio and video into a single file.

    Args:
        audio_path (str | Path): Path to the audio file.
        video_path (str | Path): Path to the video file.
        target_path (str | Path): Path to the target file.
        backend (str, optional): The backend to use for merging. Defaults to "moviepy".
    """
    assert backend in [
        "moviepy", "ffmpeg"
    ], "Backend should be moviepy or ffmpeg"
    if backend == "moviepy":
        video = VideoFileClip(video_path.__str__())
        audio = AudioFileClip(audio_path.__str__())

        video = video.with_audio(audio)

        target_path = Path(target_path)
        video.write_videofile(
            target_path,
            logger=None if not logging else "bar",
            threads=8,
            preset="ultrafast",
            ffmpeg_params=["-crf", "23"]
        )
    else:
        logging_arg = "" if logging else "-loglevel quiet"
        command = f"ffmpeg {logging_arg} -i '{video_path.__str__()}' -i '{audio_path.__str__()}' -c:v copy " \
                  f"-c:a copy -map 0:v:0 -map 1:a:0 '{target_path.__str__()}'"
        os.system(command)
