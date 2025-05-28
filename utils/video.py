from pathlib import Path
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip


def merge_audio_video(
    audio_path: str | Path, video_path: str | Path, target_path: str | Path
):
    # video = VideoFileClip(video_path.__str__())
    # audio = AudioFileClip(audio_path.__str__())
    # audio_clip = audio.subclip(0, min(audio.duration, video.duration))

    # video = video.set_audio(audio_clip)
    # target_path = Path(target_path)
    # target_path = target_path.with_stem("Y" + target_path.stem)
    # target_path = target_path.__str__().replace(" ", "_")

    # video.write_videofile(target_path)

    command = f"ffmpeg -i '{video_path.__str__()}' -i '{audio_path.__str__()}' -c:v copy " \
              f"-c:a copy -map 0:v:0 -map 1:a:0 '{target_path.__str__()}'"
    # print(command)
    os.system(command)
