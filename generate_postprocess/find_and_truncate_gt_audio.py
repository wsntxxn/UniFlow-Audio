import argparse
import os
from pathlib import Path
import subprocess

def which_ffmpeg() -> str:
    """Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    """
    result = subprocess.run(
        ["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    ffmpeg_path = result.stdout.decode("utf-8").replace("\n", "")
    return ffmpeg_path

def extract_audio_segment(video_path, output_dir, start_time, end_time, sample_rate=44100):
    """
    从视频文件中提取指定时间段的音频，并以指定采样率保存
    
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param start_time: 开始时间(秒或HH:MM:SS格式)
    :param end_time: 结束时间(秒或HH:MM:SS格式)
    :param sample_rate: 音频采样率(默认44100Hz)
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名(与视频同名，但扩展名为.wav)
    output_path = output_dir / f"{Path(video_path).stem}.wav"
    
    # 构建ffmpeg命令
    cmd = [
        '/usr/local/bin/ffmpeg',
        '-y',  # 覆盖已存在文件
        '-i', str(video_path),
        '-ss', str(start_time),  # 开始时间
        '-to', str(end_time),  # 结束时间
        '-ar', str(sample_rate),  # 采样率
        '-ac', '1',  # 立体声
        '-q:a', '0',  # 最高质量
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        print(f"成功提取音频到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"提取音频失败: {e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

def process_directory(input_dir, gt_dir,start_time, end_time, sample_rate=44100):
    """
    根据生成的所有MP4文件名，从gt_dir寻找
    
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param sample_rate: 采样率
    """
    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)

    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return
    input_dir_name = input_dir.name
    output_dir = gt_dir.parent / f"{input_dir_name}_gt_wav_{sample_rate}Hz_{start_time}s_to_{end_time}s" 
    
    # 查找所有MP4文件
    video_files = list(input_dir.glob('*.mp4'))

    # import pdb; pdb.set_trace()
    
    related_video_dir = []

    for video_file in video_files:
        gen_video = video_file.name
        gt_video = gt_dir / f"{gen_video}"
        related_video_dir.append(gt_video)

    if not related_video_dir:
        print(f"在目录 {gt_dir} 中未找到MP4文件")
        return
    
    print(f"找到 {len(video_files)} 个MP4文件，开始处理...")

    assert len(related_video_dir) == len(video_files), \
        "gt_audio samples should match gen_audio samples"
    
    for video_file in related_video_dir:
        print(f"正在处理: {video_file.name}")
        extract_audio_segment(video_file, output_dir, start_time, end_time, sample_rate)
    
    print("所有文件处理完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="包含MP4文件的输入目录")
    parser.add_argument("--gt_dir", help="保存音频文件的输出目录")
    parser.add_argument("--start_time", help="开始时间(秒或HH:MM:SS格式)")
    parser.add_argument("--end_time", help="结束时间(秒或HH:MM:SS格式)")
    parser.add_argument("--sample_rate", type=int, default=16000, 
                       help="default sample rate required by LDM-Eval")
    
    args = parser.parse_args()
    
    process_directory(
        args.input_dir,
        args.gt_dir,
        args.start_time,
        args.end_time,
        args.sample_rate
    )