import json
import re
from typing import Union, Dict
from pathlib import Path
import os

from pathlib import Path

MAX_FILE_NAME_LENGTH = 100
TASK2DATASET_CLASS = {
    't2a': "TextToAudioDataset",
    't2m': "TextToMusicDataset",
    'se': "SpeechEnhancementDataset",
    'sr': "AudioSuperResolutionDataset",
    'v2a': "VideoToAudioDataset",
    'svs': "MidiSingingDataset",
    'tts': "TextToSpeechDataset"
}


def read_jsonl_to_mapping(
    jsonl_file: Union[str, Path],
    key_col: str,
    value_col: str,
    base_path=None
) -> Dict[str, str]:
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
    TODO handle duplicate keys
    """
    mapping = {}
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            key = data[key_col]
            value = data[value_col]
            if base_path:
                value = os.path.join(base_path, value)
            mapping[key] = value
    return mapping


def sanitize_filename(name: str, max_len: int = MAX_FILE_NAME_LENGTH) -> str:
    """
    Clean and truncate a string to make it a valid and safe filename.
    """
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = name.replace('/', '_')
    max_len = min(len(name), max_len)
    return name[:max_len]


def transform_gen_fn_to_id(audio_file: Path, task: str) -> str:
    if task == "svs":
        audio_id = audio_file.stem.split("_")[0]
    elif task == "sr":
        audio_id = audio_file.stem
    elif task == "tta":
        audio_id = audio_file.stem[:11]
        # audio_id = audio_file.stem[:12] + '.wav'
    elif task == "ttm":
        audio_id = audio_file.stem[:11]
        # audio_id = audio_file.stem[:12] + '.wav'
    elif task == "v2a":
        audio_id = audio_file.stem.rsplit("_", 1)[0] + ".mp4"
    else:
        audio_id = audio_file.stem
    return audio_id


def audio_dir_to_mapping(audio_dir: str | Path, task: str) -> dict:
    mapping = {}
    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.iterdir())
    for audio_file in audio_files:
        audio_id = transform_gen_fn_to_id(audio_file, task)
        mapping[audio_id] = str(audio_file.resolve())
    return mapping
