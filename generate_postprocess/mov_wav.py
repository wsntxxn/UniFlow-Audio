import shutil
from pathlib import Path

def move_wav_files(source_dir: str, target_dir: str):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标文件夹（如果不存在）
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 移动所有 .wav 文件
    for wav_file in source_path.glob("*.wav"):
        shutil.move(str(wav_file), str(target_path / wav_file.name))
    
    print(f"Moved {len(list(target_path.glob('*.wav')))} files to {target_dir}")


move_wav_files(
    "/hpc_stor03/sjtu_home/yaoyun.zhang/work/V-AURA/inference/visualsound-2.56s", 
    "/hpc_stor03/sjtu_home/yaoyun.zhang/work/V-AURA/inference/visualsound-2.56s_gen_wav")