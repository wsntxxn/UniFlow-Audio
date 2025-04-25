from pathlib import Path
from typing import List
import json
from copy import deepcopy
import pickle

import librosa
import numpy as np
from tqdm import tqdm
from h5py import File
import sys
print(sys.path)
import os 
print(os.getcwd())
sys.path.append("./")
from utils.diffsinger_utilities import TokenTextEncoder, read_duration_from_textgrid, get_pitch

RAW_POPCS_DIR = Path("/cpfs_shared/jiahao.mei/data/svs/popcs")
TARGET_POPCS_DIR = Path("./data/popcs")
TEST_PREFIXES = [
    'popcs-说散就散',
    'popcs-隐形的翅膀',
]
SAMPLE_RATE = 24000
FRAME_SHIFT = 0.020


def load_meta_data():
    item_to_data = {}
    for song_dir in RAW_POPCS_DIR.iterdir():
        for wav_piece in song_dir.glob("*.wav"):
            segment_id = wav_piece.stem.split("_")[0]
            audio_id = f"{song_dir.name}-{wav_piece.relative_to(song_dir).stem}"
            phone_path = song_dir / f"{segment_id}_ph.txt"
            txt_path = song_dir / f"{segment_id}.txt"
            textgrid_path = song_dir / f"{segment_id}.TextGrid"
            phoneme = open(phone_path).readline()
            text = open(txt_path).readline()
            item_to_data[audio_id] = {
                "wav": wav_piece,
                "text": text,
                "phoneme": phoneme,
                "textgrid": textgrid_path
            }
    return item_to_data


def split_train_test(item_names: List[str]):
    item_names = deepcopy(item_names)
    test_item_names = [
        x
        for x in item_names if any([prefix in x for prefix in TEST_PREFIXES])
    ]
    train_item_names = [x for x in item_names if x not in set(test_item_names)]
    return train_item_names, test_item_names


def main():
    item_to_data = load_meta_data()
    item_names = sorted(list(item_to_data.keys()))
    train_item_names, test_item_names = split_train_test(item_names)
    val_item_names = test_item_names
    item_names = {
        "train": train_item_names,
        "val": val_item_names,
        "test": test_item_names
    }

    TARGET_POPCS_DIR.mkdir(parents=True, exist_ok=True)
    phone_set_file = TARGET_POPCS_DIR / "phone_set.json"
    all_phones = []
    for data in item_to_data.values():
        sentence = data["phoneme"]
        all_phones += sentence.split(' ')
    all_phones = sorted(set(all_phones))
    json.dump(all_phones, open(phone_set_file, 'w'))
    phone_tokenizer = TokenTextEncoder(
        None, vocab_list=all_phones, replace_oov=','
    )

    for split in ["train", "val", "test"]:
        (TARGET_POPCS_DIR / split).mkdir(parents=True, exist_ok=True)

        with open(TARGET_POPCS_DIR / split / "audio.jsonl", "w") as writer:
            for item_name in item_names[split]:
                writer.write(
                    json.dumps(
                        {
                            "audio_id": item_name,
                            "audio": item_to_data[item_name]["wav"].__str__(),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

        phone_pitch_file = TARGET_POPCS_DIR / split / "phone_pitch.h5"
        f0s = []

        filtered_item_names = []

        with File(phone_pitch_file, "w") as hf:
            hf.create_group("phoneme")
            hf.create_group("phoneme_duration")
            hf.create_group("f0")
            for item_name in tqdm(item_names[split]):
                raw_item = item_to_data[item_name]
                wav_file = raw_item["wav"]
                utt_duration = librosa.core.get_duration(filename=wav_file)
                phoneme = raw_item["phoneme"]
                phoneme_duration = read_duration_from_textgrid(
                    raw_item["textgrid"], phoneme, utt_duration
                )
                if phoneme_duration.min() < 0:
                    continue

                f0, _ = get_pitch(wav_file, SAMPLE_RATE, FRAME_SHIFT)
                hf["phoneme"][item_name] = phone_tokenizer.encode(phoneme)
                hf["phoneme_duration"][item_name] = np.array(phoneme_duration)
                hf["f0"][item_name] = f0

                f0s.append(f0)
                filtered_item_names.append(item_name)

        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        np.save(
            TARGET_POPCS_DIR / split / "f0_mean_std.npy",
            [np.mean(f0s).item(), np.std(f0s).item()]
        )

        with open(
            TARGET_POPCS_DIR / split / "phone_pitch.jsonl", "w"
        ) as writer:
            for item_name in filtered_item_names:
                writer.write(
                    json.dumps(
                        {
                            "audio_id": item_name,
                            "phone_pitch": phone_pitch_file.resolve().__str__()
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )


if __name__ == '__main__':
    main()
