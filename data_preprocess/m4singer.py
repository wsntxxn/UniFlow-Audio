from pathlib import Path
import json
import pickle
import argparse

TEST_PREFIXES = [
    'Alto-2#岁月神偷',
    'Alto-2#奇妙能力歌',
    'Tenor-1#一千年以后',
    'Tenor-1#童话',
    'Tenor-2#消愁',
    'Tenor-2#一荤一素',
    'Soprano-1#念奴娇赤壁怀古',
    'Soprano-1#问春',
]

parser = argparse.ArgumentParser()
parser.add_argument(
    '--raw_m4singer_dir',
    type=str,
    default="/cpfs_shared/jiahao.mei/data/svs/m4singer",
    help='path to raw m4singer data'
)
parser.add_argument(
    '--target_m4singer_dir',
    type=str,
    default="./data/m4singer",
    help='path to processed m4singer data'
)
args = parser.parse_args()
RAW_M4SINGER_DIR = Path(args.raw_m4singer_dir)
TARGET_M4SINGER_DIR = Path(args.target_m4singer_dir)


def build_spk_map(item_names, item_to_data):
    spk_map = set()
    for item_name in item_names:
        spk_name = item_to_data[item_name]["spk"]
        spk_map.add(spk_name)
    spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
    return spk_map


def main():

    TARGET_M4SINGER_DIR.mkdir(parents=True, exist_ok=True)

    item_to_data = {}

    song_items = json.load(open(RAW_M4SINGER_DIR / "meta.json"))
    for song_item in song_items:
        item_name = song_item['item_name']
        singer, song_name, sent_id = item_name.split("#")
        item_to_data[item_name] = {
            "wav": f'{RAW_M4SINGER_DIR}/{singer}#{song_name}/{sent_id}.wav',
            "txt": song_item['txt'],
            "phoneme": ' '.join(song_item['phs']),
            "phoneme_duration": song_item['ph_dur'],
            "midi": song_item['notes'],
            "midi_duration": song_item['notes_dur'],
            "is_slur": song_item['is_slur'],
            "spk": singer
        }

    # train / test split
    item_names = sorted(list(item_to_data.keys()))
    test_item_names = [
        x
        for x in item_names if any([x.startswith(ts) for ts in TEST_PREFIXES])
    ]
    val_item_names = test_item_names
    train_item_names = [x for x in item_names if x not in set(test_item_names)]
    split_to_item_names = {
        "train": train_item_names,
        "val": val_item_names,
        "test": test_item_names
    }

    # speaker mapping
    spks = [x["spk"] for x in item_to_data.values()]
    print('spkers: ', set(spks))
    spk_map = build_spk_map(item_names, item_to_data)
    json.dump(
        list(spk_map.keys()), open(TARGET_M4SINGER_DIR / "spk_set.json", 'w')
    )

    # phoneme tokenizer
    all_phones = []
    for data in item_to_data.values():
        sentence = data["phoneme"]
        all_phones += sentence.split(' ')
    all_phones = sorted(set(all_phones))
    json.dump(all_phones, open(TARGET_M4SINGER_DIR / "phone_set.json", 'w'))
    print("build phone set: ", all_phones)

    for split in ["train", "val", "test"]:
        (TARGET_M4SINGER_DIR / split).mkdir(parents=True, exist_ok=True)

        with open(TARGET_M4SINGER_DIR / split / "audio.jsonl", "w") as writer:
            for item_name in split_to_item_names[split]:
                writer.write(
                    json.dumps(
                        {
                            "audio_id": item_name,
                            "audio": item_to_data[item_name]["wav"].__str__()
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

        midi_file = TARGET_M4SINGER_DIR / split / "midi.pkl"
        midi_data = {}
        for item_name in split_to_item_names[split]:
            raw_item = item_to_data[item_name]
            item_data = {
                "phoneme": raw_item["phoneme"],
                "phoneme_duration": raw_item["phoneme_duration"],
                "midi": raw_item["midi"],
                "midi_duration": raw_item["midi_duration"],
                "is_slur": raw_item["is_slur"],
                "spk": raw_item["spk"],
                "text": raw_item["txt"],
            }
            midi_data[item_name] = item_data
        pickle.dump(midi_data, open(midi_file, "wb"))

        with open(TARGET_M4SINGER_DIR / split / "midi.jsonl", "w") as writer:
            for item_name in split_to_item_names[split]:
                writer.write(
                    json.dumps(
                        {
                            "audio_id": item_name,
                            "midi": midi_file.resolve().__str__()
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )


if __name__ == '__main__':
    main()
