"""
Description: 
    This script processes an audio dataset by performing the following tasks:
    - Splits the dataset into train, validation, and test sets.
    - Resamples audio to 24kHz and applies random lowpass filters.
    - Saves processed audio files into structured directories.
    - the raw audio will save into raw folder(24k) and lowpass will save to low folder (2-4k)

Usage:
    
    ```
    python sr_24k.py --input_folder /path/to/input \
                     --output_folder /path/to/output \
                     --dataset_name (moises,musdb,HQtts)
    ```

    for dataset provided, the input_folder should be "raw" of each dataset, like "HQ-TTS/raw"
"""

import os
import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
import random
import argparse
import json
import uuid
import hashlib
import posixpath
import shutil


def random_lowpass_filter():
    filter_types = ['chebyshev', 'elliptic', 'butterworth', 'boxcar']
    filter_type = random.choice(filter_types)
    cutoff_freq = random.uniform(2000, 4000)  # 48k截止频率在 2kHz 到 16kHz
    order = random.randint(2, 10)
    return filter_type, cutoff_freq, order


def apply_lowpass_filter(signal_data, sr, filter_type, cutoff_freq, order):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    if filter_type == 'chebyshev':
        b, a = signal.cheby1(order, 0.5, normal_cutoff, btype='low')
    elif filter_type == 'elliptic':
        b, a = signal.ellip(order, 0.5, 40, normal_cutoff, btype='low')
    elif filter_type == 'butterworth':
        b, a = signal.butter(order, normal_cutoff, btype='low')
    elif filter_type == 'boxcar':
        b = np.ones(order) / order
        a = 1

    return signal.filtfilt(b, a, signal_data, axis=-1)


def generate_unique_audio_id(filepath, dataset_name):
    hash_md5 = hashlib.md5(filepath.encode()).hexdigest()
    return f"{dataset_name}_{hash_md5[:10]}_{uuid.uuid4().hex[:8]}"


def collect_audio_files(input_dir):
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.wav', '.flac')):
                full_path = os.path.join(root, filename)
                audio_files.append(full_path)
    return audio_files


def split_files(file_list, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    return train_files, val_files, test_files


def process_partition(files, input_dir, output_partition_dir, dataset_name):
    raw_dir = os.path.join(output_partition_dir, "raw")
    low_dir = os.path.join(output_partition_dir, "low")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(low_dir, exist_ok=True)

    content_jsonl_path = os.path.join(output_partition_dir, "content.jsonl")
    audio_jsonl_path = os.path.join(output_partition_dir, "audio.jsonl")

    content_entries = []
    audio_entries = []

    for file_path in files:
        try:
            rel_path = os.path.relpath(file_path, input_dir)

            raw_target_base = os.path.join(raw_dir, rel_path)
            low_target_base = os.path.join(low_dir, rel_path)

            os.makedirs(os.path.dirname(raw_target_base), exist_ok=True)
            os.makedirs(os.path.dirname(low_target_base), exist_ok=True)

            audio, sr = librosa.load(file_path, sr=24000, mono=False)

            duration = audio.shape[-1] / sr

            if duration > 20:
                num_clips = int(duration // 10) + 1  # clip
                for i in range(num_clips):
                    start = i * 10 * sr
                    end = (i + 1) * 10 * sr
                    if end > audio.shape[-1]:
                        end = audio.shape[-1]

                    clip = audio[..., start:end]

                    if clip.shape[-1] < 2 * sr:
                        continue

                    clip_rel_path = os.path.splitext(rel_path)[
                        0] + f"_clip{i+1}" + os.path.splitext(rel_path)[1]
                    raw_target = os.path.abspath(
                        os.path.join(raw_dir, clip_rel_path)
                    )
                    low_target = os.path.abspath(
                        os.path.join(low_dir, clip_rel_path)
                    )

                    if clip.ndim == 1:
                        audio_to_save = clip
                    else:
                        audio_to_save = clip.T
                    sf.write(raw_target, audio_to_save, 24000)

                    filter_type, cutoff_freq, order = random_lowpass_filter()
                    low_res_audio = apply_lowpass_filter(
                        clip,
                        sr=24000,
                        filter_type=filter_type,
                        cutoff_freq=cutoff_freq,
                        order=order
                    )

                    if low_res_audio.ndim == 1:
                        low_audio_to_save = low_res_audio
                    else:
                        low_audio_to_save = low_res_audio.T
                    sf.write(low_target, low_audio_to_save, 24000)

                    unique_id = generate_unique_audio_id(
                        posixpath.join(*posixpath.split(clip_rel_path)),
                        dataset_name
                    )

                    # **使用绝对路径**
                    audio_entry = {"audio_id": unique_id, "audio": raw_target}
                    content_entry = {
                        "audio_id": unique_id,
                        "caption": low_target
                    }

                    audio_entries.append(audio_entry)
                    content_entries.append(content_entry)

                    print(
                        f"Processed: {file_path} -> raw: {raw_target} | low: {low_target}"
                    )
            else:

                if audio.ndim == 1:
                    audio_to_save = audio
                else:
                    audio_to_save = audio.T
                sf.write(raw_target_base, audio_to_save, 24000)

                filter_type, cutoff_freq, order = random_lowpass_filter()
                low_res_audio = apply_lowpass_filter(
                    audio,
                    sr=24000,
                    filter_type=filter_type,
                    cutoff_freq=cutoff_freq,
                    order=order
                )

                if low_res_audio.ndim == 1:
                    low_audio_to_save = low_res_audio
                else:
                    low_audio_to_save = low_res_audio.T
                sf.write(low_target_base, low_audio_to_save, 24000)

                unique_id = generate_unique_audio_id(
                    posixpath.join(*posixpath.split(rel_path)), dataset_name
                )

                audio_entry = {"audio_id": unique_id, "audio": raw_target_base}
                content_entry = {
                    "audio_id": unique_id,
                    "caption": low_target_base
                }

                audio_entries.append(audio_entry)
                content_entries.append(content_entry)

                print(
                    f"Processed: {file_path} -> raw: {raw_target_base} | low: {low_target_base}"
                )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    #
    with open(audio_jsonl_path, "w", encoding="utf-8") as f:
        for entry in audio_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(content_jsonl_path, "w", encoding="utf-8") as f:
        for entry in content_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"Saved {len(audio_entries)} entries to {audio_jsonl_path} and {content_jsonl_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=
        "Process audio dataset: resample to 24kHz and apply lowpass filter (2-4kHz cutoff). "
        "Split input into train/val/test and output to a new directory without modifying "
        "the original dataset."
    )
    parser.add_argument(
        '--input_folder',
        "-i",
        type=str,
        required=True,
        help='Input folder containing original audio files (can be nested)'
    )
    parser.add_argument(
        '--output_folder',
        "-o",
        type=str,
        required=True,
        help='Output folder to create new dataset with train/val/test splits'
    )
    parser.add_argument(
        '--dataset_name',
        "-d",
        type=str,
        required=True,
        help='Dataset name for unique ID generation'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training samples'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Ratio of validation samples'
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    dataset_name = args.dataset_name

    os.makedirs(output_folder, exist_ok=True)
    all_files = collect_audio_files(input_folder)
    print(f"Found {len(all_files)} audio files in {input_folder}")

    # 分割为 train, val, test
    train_files, val_files, test_files = split_files(
        all_files, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    print(
        f"Split into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files"
    )

    for partition, files in zip(
        ["train", "val", "test"], [train_files, val_files, test_files]
    ):
        partition_output = os.path.join(output_folder, partition)
        os.makedirs(partition_output, exist_ok=True)
        print(
            f"Processing partition {partition} with {len(files)} files, output to {partition_output}"
        )
        process_partition(files, input_folder, partition_output, dataset_name)


if __name__ == '__main__':
    main()
