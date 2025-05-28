import json
import re
from pathlib import Path

MAX_FILE_NAME_LENGTH = 50


def read_jsonl_to_mapping(
    jsonl_file: str | Path, key_col: str, value_col: str
) -> dict[str, str]:
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
