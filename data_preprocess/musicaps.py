import json
import random
from pathlib import Path
from tqdm import tqdm

# 参数配置
input_jsonl = "/cpfs_shared/jiahao.mei/data/ttm/MusicCaps/metadata/MusicCaps.jsonl"  # 替换成你的输入路径
output_dir = Path("/cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/musiccaps")   # 替换成你的输出根目录
splits = {"train": 0.8, "val": 0.1, "test": 0.1}

# 创建输出目录
for split in splits:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

# 读取所有数据
with open(input_jsonl, "r") as f:
    data = [json.loads(line) for line in f]

# 打乱数据
random.shuffle(data)

# 按比例划分数据
total = len(data)
train_end = int(total * splits["train"])
val_end = train_end + int(total * splits["val"])

split_data = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:]
}

# 写入新的jsonl
for split, items in split_data.items():
    audio_lines = []
    caption_lines = []
    for item in tqdm(items, desc=f"Processing {split}"):
        # 获取路径并提取音频文件名
        wav_path = item["wavPath"]
        # audio_id = Path(wav_path).name.replace(".mp3", ".wav")  # 替换扩展名为 .wav
        audio_id=item["uuid"]
        # 生成 audio.jsonl 条目
        audio_lines.append(json.dumps({
            "audio_id": audio_id,
            "audio": wav_path
        }))

        # 生成 caption.jsonl 条目
        caption_lines.append(json.dumps({
            "audio_id": audio_id,
            "caption": item["inputData"]
        }))

    # 保存到对应子文件夹
    with open(output_dir / split / "audio.jsonl", "w") as fa:
        fa.write("\n".join(audio_lines) + "\n")
    with open(output_dir / split / "caption.jsonl", "w") as fc:
        fc.write("\n".join(caption_lines) + "\n")

print("✅ 数据划分并保存完成！")