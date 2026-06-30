import re
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
    elif task == "t2a":
        audio_id = audio_file.stem[:11]
        # audio_id = audio_file.stem[:12] + '.wav'
    elif task == "t2m":
        audio_id = audio_file.stem[:11]
        # audio_id = audio_file.stem[:12] + '.wav'
    elif task == "v2a":
        fname = audio_file.stem
        yid = fname[:11]
        start = fname[12:].split("_")[0]
        audio_id = f"{yid}_{start}.mp4"
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
