from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip


def merge_audio_video(audio_path, video_path, target_path):
    video = VideoFileClip(video_path.__str__())

    audio = AudioFileClip(audio_path)
    audio_clip = audio.subclip(0, min(audio.duration, video.duration))

    video = video.set_audio(audio_clip)
    video.write_videofile(target_path)
