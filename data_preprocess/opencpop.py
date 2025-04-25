from pathlib import Path
import json
import pickle

import librosa

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.diffsinger_utilities import TokenTextEncoder

RAW_OPENCPOP_DIR = Path(
    "/cpfs_shared/jiahao.mei/data/svs/opencpop/segments"
)
TARGET_OPENCPOP_DIR = Path("./data/opencpop")


def main():

    item_to_data = {}

    with open(RAW_OPENCPOP_DIR / "transcriptions.txt") as reader:
        for utterance_label in reader.readlines():
            utterance_label = utterance_label.strip()
            song_info = utterance_label.split('|')
            item_name = song_info[0]
            item_to_data[item_name] = {
                "wav": RAW_OPENCPOP_DIR / f"wavs/{item_name}.wav",
                "txt": song_info[1],
                "phoneme": song_info[2],
                "phoneme_duration": [
                    float(x) for x in song_info[5].split(" ")
                ],
                "midi":
                    [
                        librosa.note_to_midi(x.split("/")[0])
                        if x != 'rest' else 0 for x in song_info[3].split(" ")
                    ],
                "midi_duration": [float(x) for x in song_info[4].split(" ")],
                "is_slur": [int(x) for x in song_info[6].split(" ")],
                "spk": 'opencpop'
            }

    item_names = sorted(list(item_to_data.keys()))

    test_item_names = []
    with open(RAW_OPENCPOP_DIR / "test.txt") as reader:
        for line in reader.readlines():
            test_item_names.append(line.strip().split("|")[0])
    val_item_names = test_item_names
    train_item_names = [x for x in item_names if x not in set(test_item_names)]

    item_names = {
        "train": train_item_names,
        "val": val_item_names,
        "test": test_item_names
    }

    TARGET_OPENCPOP_DIR.mkdir(parents=True, exist_ok=True)
    phone_set_file = TARGET_OPENCPOP_DIR / "phone_set.json"
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
        (TARGET_OPENCPOP_DIR / split).mkdir(parents=True, exist_ok=True)

        with open(TARGET_OPENCPOP_DIR / split / "audio.jsonl", "w") as writer:
            for item_name in item_names[split]:
                writer.write(
                    json.dumps(
                        {
                            "audio_id": item_name,
                            "audio": item_to_data[item_name]["wav"].__str__()
                        }
                    ) + "\n"
                )

        midi_file = TARGET_OPENCPOP_DIR / split / "midi.pkl"
        midi_data = {}
        for item_name in item_names[split]:
            raw_item = item_to_data[item_name]
            item_data = {
                "phoneme": raw_item["phoneme"],
                "phoneme_duration": raw_item["phoneme_duration"],
                "midi": raw_item["midi"],
                "midi_duration": raw_item["midi_duration"],
                "is_slur": raw_item["is_slur"],
            }
            midi_data[item_name] = item_data
        pickle.dump(midi_data, open(midi_file, "wb"))

        with open(TARGET_OPENCPOP_DIR / split / "midi.jsonl", "w") as writer:
            for item_name in item_names[split]:
                writer.write(
                    json.dumps(
                        {
                            "audio_id": item_name,
                            "midi": midi_file.resolve().__str__()
                        }
                    ) + "\n"
                )


if __name__ == '__main__':
    main()
