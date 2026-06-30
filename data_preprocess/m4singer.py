import json
import fire
import pickle
from pathlib import Path
import librosa
from tqdm import tqdm

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

CONSONANTS = [
    'b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r',
    's', 'sh', 't', 'x', 'z', 'zh'
]


def build_spk_map(item_names, item_to_data):
    spk_map = set()
    for item_name in item_names:
        spk_name = item_to_data[item_name]["spk"]
        spk_map.add(spk_name)
    spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
    return spk_map


class Runner:
    def prepare_diffsinger_style(
        self,
        raw_m4singer_dir: str,
        target_m4singer_dir: str = "./data/m4singer"
    ):

        target_m4singer_dir.mkdir(parents=True, exist_ok=True)

        item_to_data = {}

        song_items = json.load(open(raw_m4singer_dir / "meta.json"))
        for song_item in song_items:
            item_name = song_item['item_name']
            singer, song_name, sent_id = item_name.split("#")
            item_to_data[item_name] = {
                "wav":
                    f'{raw_m4singer_dir}/{singer}#{song_name}/{sent_id}.wav',
                "txt":
                    song_item['txt'],
                "phoneme":
                    ' '.join(song_item['phs']),
                "phoneme_duration":
                    song_item['ph_dur'],
                "midi":
                    song_item['notes'],
                "midi_duration":
                    song_item['notes_dur'],
                "is_slur":
                    song_item['is_slur'],
                "spk":
                    singer
            }

        # train / test split
        item_names = sorted(list(item_to_data.keys()))
        test_item_names = [
            x for x in item_names
            if any([x.startswith(ts) for ts in TEST_PREFIXES])
        ]
        val_item_names = test_item_names
        train_item_names = [
            x for x in item_names if x not in set(test_item_names)
        ]
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
            list(spk_map.keys()),
            open(target_m4singer_dir / "spk_set.json", 'w')
        )

        # phoneme tokenizer
        all_phones = []
        for data in item_to_data.values():
            sentence = data["phoneme"]
            all_phones += sentence.split(' ')
        all_phones = sorted(set(all_phones))
        json.dump(
            all_phones, open(target_m4singer_dir / "phone_set.json", 'w')
        )
        print("build phone set: ", all_phones)

        for split in ["train", "val", "test"]:
            (target_m4singer_dir / split).mkdir(parents=True, exist_ok=True)

            with open(
                target_m4singer_dir / split / "audio.jsonl", "w"
            ) as writer:
                for item_name in split_to_item_names[split]:
                    writer.write(
                        json.dumps(
                            {
                                "audio_id":
                                    item_name,
                                "audio":
                                    item_to_data[item_name]["wav"].__str__()
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )

            midi_file = target_m4singer_dir / split / "midi.pkl"
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

            with open(
                target_m4singer_dir / split / "midi.jsonl", "w"
            ) as writer:
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

    def prepare_prompt_length(
        self, data_dir: str = "./data/m4singer", max_prompt_frac: float = 0.7
    ):
        data_dir = Path(data_dir)

        for split in ["train", "val"]:
            prompt_length_file = data_dir / split / "prompt_length.json"
            out_data = {}
            with open(prompt_length_file, "w") as writer, \
                open(data_dir / split / "midi.pkl", "rb") as reader:
                data = pickle.load(reader)
                for audio_id in data:
                    phoneme_durations = data[audio_id]["phoneme_duration"]
                    phonemes = data[audio_id]["phoneme"].split()

                    tot_duration = sum(phoneme_durations)
                    if tot_duration > 3.0:
                        duration_threshold = 3.0
                    else:
                        duration_threshold = tot_duration / 2

                    cur_duration = 0.0
                    for idx, (phoneme, duration) in enumerate(
                        zip(phonemes, phoneme_durations)
                    ):
                        if cur_duration + duration > duration_threshold:
                            if idx < len(phonemes) - 1 and phonemes[
                                idx + 1] in CONSONANTS:
                                break
                        cur_duration += duration

                    if idx > len(phonemes) * max_prompt_frac:
                        res = len(phonemes) // 2
                    else:
                        res = idx + 1

                    res = min(len(phonemes) - 1, res)

                    out_data[audio_id] = res

            json.dump(
                out_data,
                open(prompt_length_file, "w"),
                indent=2,
                ensure_ascii=False
            )

    def prepare_cross_sentence_test(self, data_dir: str = "./data/m4singer"):
        data_dir = Path(data_dir)
        audio_jsonl = data_dir / "test" / "audio.jsonl"
        midi_jsonl = data_dir / "test" / "midi.jsonl"
        out_jsonl = data_dir / "test" / "midi_cross_sentence.jsonl"

        aid_to_duration = {}
        aid_to_audio = {}
        song_to_aids = {}

        with open(audio_jsonl, "r") as reader:
            for line in tqdm(reader.readlines()):
                item = json.loads(line)
                audio_id = item["audio_id"]
                audio_path = item["audio"]
                duration = librosa.core.get_duration(path=audio_path)
                aid_to_duration[audio_id] = duration
                aid_to_audio[audio_id] = audio_path
                song = audio_id.rsplit("#", 1)[0]
                if song not in song_to_aids:
                    song_to_aids[song] = []
                song_to_aids[song].append(audio_id)

        song_to_prompt = {}
        for song, aids in song_to_aids.items():
            song_to_prompt[song] = []
            for aid in aids:
                if aid_to_duration[aid] >= 2.0 and aid_to_duration[aid] <= 5.0:
                    song_to_prompt[song].append(aid)
                    if len(song_to_prompt[song]) == 2:
                        break

            if len(song_to_prompt[song]) < 2:
                song_aids = {aid: aid_to_duration[aid] for aid in aids}
                song_aids = sorted(
                    song_aids.items(), key=lambda x: x[1], reverse=True
                )
                song_to_prompt[song] = [song_aids[0][0], song_aids[1][0]]

        # print(song_to_prompt)
        out_data = []
        with open(midi_jsonl, "r") as reader:
            for line in tqdm(reader.readlines()):
                item = json.loads(line)
                audio_id = item["audio_id"]
                song = audio_id.rsplit("#", 1)[0]
                if audio_id == song_to_prompt[song][0]:
                    prompt_audio = song_to_prompt[song][1]
                else:
                    prompt_audio = song_to_prompt[song][0]
                item["prompt_audio"] = aid_to_audio[prompt_audio]
                out_data.append(item)

        with open(out_jsonl, "w") as writer:
            for item in out_data:
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    fire.Fire(Runner)
