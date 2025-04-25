import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import os


def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def save_metadata(file_path, metadata):
    with open(file_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')


def update_paths(metadata, base_dir, key):
    for item in metadata:
        item[key] = str(Path(base_dir) / item[key])
    return metadata


def split_and_save_metadata(
    base_dirs, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
):
    combined_split_data = {
        'train': {
            'audio': [],
            'caption': [],
            'condition': []
        },
        'val': {
            'audio': [],
            'caption': [],
            'condition': []
        },
        'test': {
            'audio': [],
            'caption': [],
            'condition': []
        }
    }

    for base_dir in base_dirs:
        # 加载元数据
        audio_metadata = load_metadata(base_dir / 'metadata_audio.jsonl')
        caption_metadata = load_metadata(base_dir / 'metadata_caption.jsonl')
        condition_metadata = load_metadata(
            base_dir / 'metadata_condition.jsonl'
        )

        # 划分元数据
        train_audio, temp_audio = train_test_split(
            audio_metadata, train_size=train_ratio, random_state=42
        )
        val_audio, test_audio = train_test_split(
            temp_audio,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=42
        )

        # 确保不同元数据文件的划分一致
        splits = {
            'train': set(item['UUID'] for item in train_audio),
            'val': set(item['UUID'] for item in val_audio),
            'test': set(item['UUID'] for item in test_audio)
        }

        split_data = {
            split: {
                'audio': [],
                'caption': [],
                'condition': []
            }
            for split in splits
        }

        for item in caption_metadata:
            for split, uuids in splits.items():
                if item['UUID'] in uuids:
                    split_data[split]['caption'].append(item)

        for item in condition_metadata:
            for split, uuids in splits.items():
                if item['UUID'] in uuids:
                    split_data[split]['condition'].append(item)

        # 更新为绝对路径
        for split in ['train', 'val', 'test']:
            split_data[split]['audio'] = update_paths(
                [
                    item
                    for item in audio_metadata if item['UUID'] in splits[split]
                ], base_dir, 'WavPath'
            )
            split_data[split]['caption'] = update_paths(
                split_data[split]['caption'], base_dir, 'InputPath'
            )
            split_data[split]['condition'] = update_paths(
                split_data[split]['condition'], base_dir, 'ConditionPath'
            )

        # 合并所有 base_dir 的数据
        for split in ['train', 'val', 'test']:
            combined_split_data[split]['audio'].extend(
                split_data[split]['audio']
            )
            combined_split_data[split]['caption'].extend(
                split_data[split]['caption']
            )
            combined_split_data[split]['condition'].extend(
                split_data[split]['condition']
            )

    # 创建目标目录并保存更新后的元数据
    for split in ['train', 'val', 'test']:
        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        save_metadata(
            split_dir / 'metadata_audio.jsonl',
            combined_split_data[split]['audio']
        )
        save_metadata(
            split_dir / 'metadata_caption.jsonl',
            combined_split_data[split]['caption']
        )
        save_metadata(
            split_dir / 'metadata_condition.jsonl',
            combined_split_data[split]['condition']
        )


def append_metadata(source_folder, target_folder):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)

    # 处理测试集元数据
    test_audio_metadata = load_metadata(
        source_folder / 'test_metadata_audio.jsonl'
    )
    test_caption_metadata = load_metadata(
        source_folder / 'test_metadata_caption.jsonl'
    )
    test_target_audio_path = target_folder / 'test' / 'metadata_audio.jsonl'
    test_target_caption_path = target_folder / 'test' / 'metadata_caption.jsonl'

    # 更新路径为绝对路径
    test_audio_metadata = update_paths(
        test_audio_metadata, source_folder, 'WavPath'
    )
    test_caption_metadata = update_paths(
        test_caption_metadata, source_folder, 'InputPath'
    )

    existing_test_audio = load_metadata(
        test_target_audio_path
    ) if test_target_audio_path.exists() else []
    existing_test_caption = load_metadata(
        test_target_caption_path
    ) if test_target_caption_path.exists() else []

    existing_test_audio.extend(test_audio_metadata)
    existing_test_caption.extend(test_caption_metadata)

    save_metadata(test_target_audio_path, existing_test_audio)
    save_metadata(test_target_caption_path, existing_test_caption)

    # 处理训练集元数据
    train_audio_metadata = load_metadata(
        source_folder / 'train_metadata_audio.jsonl'
    )
    train_caption_metadata = load_metadata(
        source_folder / 'train_metadata_caption.jsonl'
    )
    train_target_audio_path = target_folder / 'train' / 'metadata_audio.jsonl'
    train_target_caption_path = target_folder / 'train' / 'metadata_caption.jsonl'

    # 更新路径为绝对路径
    train_audio_metadata = update_paths(
        train_audio_metadata, source_folder, 'WavPath'
    )
    train_caption_metadata = update_paths(
        train_caption_metadata, source_folder, 'InputPath'
    )

    existing_train_audio = load_metadata(
        train_target_audio_path
    ) if train_target_audio_path.exists() else []
    existing_train_caption = load_metadata(
        train_target_caption_path
    ) if train_target_caption_path.exists() else []

    existing_train_audio.extend(train_audio_metadata)
    existing_train_caption.extend(train_caption_metadata)

    save_metadata(train_target_audio_path, existing_train_audio)
    save_metadata(train_target_caption_path, existing_train_caption)


if __name__ == "__main__":
    # 多个原目录，包含 matadata 文件，包含除voicebank+demand之外的se数据集
    base_dirs = [
        # Path("/cpfs_shared/jiahao.mei/data/se/LJSpeech+Musan"),
        # Path("/cpfs_shared/jiahao.mei/data/se/Libritts+Wham"),
        Path("/cpfs_shared/jiahao.mei/data/se/VCTK+Wham"),
        # Path("/hpc_stor03/sjtu_home/zihao.zheng/data/Libritts_360")
    ]
    # 目标目录，保存训练、验证、测试集的 metadata 文件
    target_dir = Path(
        f"/cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/VCTK+Wham"
    )
    split_and_save_metadata(base_dirs, target_dir)
    print("Done!")
    # # voicebank+demand 数据集需要单独处理
    # add_dir = "/cpfs_shared/jiahao.mei/data/se/VCTK+Demand"
    # append_metadata(add_dir, target_dir)
    # print("vctk+demand done!")
