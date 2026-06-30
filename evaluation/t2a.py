#!/usr/bin/env python3
"""
Text-to-Audio evaluation script.

This script evaluates generated audio against reference audio using metrics
including FAD, FD, KL, and CLAP scores.
"""

import argparse
from collections import defaultdict
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from accel_hydra.utils.general import read_jsonl_to_mapping
import laion_clap

# Ref: https://github.com/haoheliu/audioldm_eval/tree/main
# This script uses a locally modified version of audioldm_eval.
from audioldm_eval import EvaluationHelper

from utils.general import audio_dir_to_mapping

torch.multiprocessing.set_sharing_strategy('file_system')


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    else:
        device = "cpu"
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def dist_barrier():
    if dist.is_initialized():
        dist.barrier()


def compute_clap_metrics(batch: dict, model: laion_clap.CLAP_Module):

    with torch.no_grad():
        text_embed = model.get_text_embedding(batch["text"], use_tensor=False)
        audio_embed = model.get_audio_embedding_from_data(
            x=batch["audio"], use_tensor=False
        )
        audio_norm = np.linalg.norm(audio_embed, axis=1)
        text_norm = np.linalg.norm(text_embed, axis=1)
        clap_sim = np.sum(audio_embed * text_embed,
                          axis=1) / (audio_norm * text_norm)

    return clap_sim


class AudioTextDataset(torch.utils.data.Dataset):
    def __init__(self, ref_aid_to_captions: dict, gen_aid_to_audios: dict):
        self.ref_aid_to_captions = ref_aid_to_captions
        self.gen_aid_to_audios = gen_aid_to_audios
        self.audio_ids = list(ref_aid_to_captions.keys())

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, index):
        audio_id = self.audio_ids[index]
        caption = self.ref_aid_to_captions[audio_id]
        gen_audio = self.gen_aid_to_audios[audio_id]
        waveform, _ = librosa.load(gen_audio, sr=48000)
        return {
            "audio_id": audio_id,
            "audio": waveform,
            "text": caption,
        }

    def collate_fn(self, batch):
        return {
            "audio_id": [item["audio_id"] for item in batch],
            "audio": [item["audio"] for item in batch],
            "text": [item["text"] for item in batch],
        }


def get_common_folder_path(audio_dict):
    """
    Extract the common folder path from audio path dictionary.
    
    Parameters:
    audio_dict -- Dictionary in format {audio_id: audio_path}
    
    Returns:
    common_folder -- Common folder path (None if no common path)
    is_same_folder -- Boolean indicating if all audios are in the same folder
    """
    if not audio_dict:
        return None, False
    paths = list(audio_dict.values())
    parent_folders = [os.path.dirname(path) for path in paths]
    common_prefix = str(Path(os.path.commonpath(parent_folders)).resolve())
    is_same_folder = all(
        parent == parent_folders[0] for parent in parent_folders
    )

    return common_prefix, is_same_folder


def evaluate(args):
    """Calculate FAD, FD, KL, etc. socres."""
    rank, world_size, device = setup_distributed()
    is_main_process = rank == 0

    ref_aid_to_audios = read_jsonl_to_mapping(
        args.ref_audio_jsonl,
        "audio_id",
        "audio",
    )
    if args.gen_audio_jsonl is not None:
        gen_aid_to_audios = read_jsonl_to_mapping(
            args.gen_audio_jsonl, "audio_id", "audio"
        )
    elif args.gen_audio_dir is not None:
        gen_aid_to_audios = audio_dir_to_mapping(args.gen_audio_dir, args.task)

    keys = deepcopy(list(ref_aid_to_audios.keys()))
    for key in keys:
        if key not in gen_aid_to_audios:
            ref_aid_to_audios.pop(key)
    """Calculate ldm eval score: FAD, FD, KL score"""
    args.device = device
    backbone = "cnn14" if args.task == "t2a" else "mert"
    evaluator = EvaluationHelper(16000, args.device, backbone=backbone)

    gen_folder_path, gen_is_same_folder = get_common_folder_path(
        gen_aid_to_audios
    )
    ref_folder_path, ref_is_same_folder = get_common_folder_path(
        ref_aid_to_audios
    )
    assert gen_is_same_folder == True, "Generated audio files must be in the same folder."
    assert ref_is_same_folder == True, "Reference audio files must be in the same folder."

    eval_result = evaluator.main(
        gen_aid_to_audios,
        ref_aid_to_audios,
        recalculate=args.recalculate,
        num_workers=args.num_workers,
    )

    assert ref_aid_to_audios.keys() == gen_aid_to_audios.keys(
    ), "Reference and generated audio IDs do not match"

    results = defaultdict(dict)
    results.update(eval_result)
    """The CLAP calculation still needs to be verified."""

    ref_aid_to_captions = read_jsonl_to_mapping(
        args.ref_caption_jsonl, "audio_id", "caption"
    )

    dataset = AudioTextDataset(ref_aid_to_captions, gen_aid_to_audios)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if dist.is_initialized() else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn
    )
    clap_scorer = laion_clap.CLAP_Module(enable_fusion=False)
    # If CLAP fails to load, set verbose=False to True to check errors.
    clap_model_path = os.environ["CLAP_MODEL_PATH"]
    assert clap_model_path is not None, "CLAP_MODEL_PATH environment variable not set."
    clap_scorer.load_ckpt(ckpt=clap_model_path, verbose=False)
    clap_scorer.eval()
    clap_scores = {}
    for batch in tqdm(
        dataloader,
        desc="Computing CLAP score",
        disable=dist.is_initialized() and rank != 0,
    ):
        scores = compute_clap_metrics(batch, clap_scorer)
        for audio_id, score in zip(batch["audio_id"], scores):
            clap_scores[audio_id] = score.item()

    if dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, clap_scores)
        clap_scores = {}
        for part in gathered:
            clap_scores.update(part or {})
    if is_main_process:
        results["CLAP_score"].update(clap_scores)
        with open(args.output_file, "w") as writer:
            for metric, values in results.items():
                if metric == "CLAP_score":
                    print_msg = f"{metric}: {np.mean(list(values.values())):.3f}"
                    print(print_msg)
                    print(print_msg, file=writer)
                    if args.clap_per_audio:
                        for audio_id, score in values.items():
                            score_msg = f"{audio_id}: {score:.3f}"
                            print(score_msg, file=writer)

                else:
                    print_msg = f"{metric}: {values:.3f}"
                    print(print_msg)
                    print(print_msg, file=writer)

    dist_barrier()
    cleanup_distributed()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=
        "Evaluate text-to-audio generation using FAD, FD, KL, and CLAP metrics.",
        epilog="""
Examples:
    export CLAP_MODEL_PATH="xxx/630k-audioset-best.pt"
    python evaluation/t2a.py \\
      --ref_audio_jsonl data/audiocaps_v2/test/audio.jsonl \\
      --rc data/audiocaps_v2/test/caption.jsonl \\
      --gd xxxx/t2a_infer \\
      -o xxx/t2a.txt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ref_audio_jsonl",
        "-r",
        type=str,
        required=True,
        help="path to reference audio jsonl file"
    )
    parser.add_argument(
        "--ref_caption_jsonl",
        "-rc",
        type=str,
        required=True,
        help="path to reference caption jsonl file"
    )
    parser.add_argument(
        "--gen_audio_dir",
        "-gd",
        type=str,
        help="path to generated audio directory"
    )
    parser.add_argument(
        "--gen_audio_jsonl",
        "-gj",
        type=str,
        help="path to generated audio jsonl file"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="path to output file"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="t2a",
        help="task type, text-to-audio (t2a) or text_to_music (t2m)",
        choices=["t2a", "t2m"]
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        default=4,
        type=int,
        help="number of workers for parallel processing"
    )
    parser.add_argument(
        "--clap_per_audio",
        "-p",
        action="store_true",
        help="calculate and store CLAP score for each audio clip"
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="recalculate embeddings for metric scores"
    )

    args = parser.parse_args()

    evaluate(args)
