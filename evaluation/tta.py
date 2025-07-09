# Usage:
# export CLAP_MODEL_PATH=/cpfs02/shared/speechllm/xuxuenan/hf_cache/hub/models--lukewys--laion_clap/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/630k-audioset-best.pt
# python evaluation/tta.py \
#   --ref_audio_jsonl /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/audio.jsonl \
#   --ref_caption_jsonl  /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/caption.jsonl \
#   --gen_audio_dir /cpfs_shared/jiahao.mei/code/x_to_audio_generation/xxx/tta_infer \
#   --output_file evaluation/result/tta.jsonl \
#   -c 16

# gt
# python evaluation/tta.py \
#   --ref_audio_jsonl /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/audio.jsonl \
#   --ref_caption_jsonl  /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/caption.jsonl \
#   --gen_audio_dir /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/audio \
#   --output_file evaluation/result/tta.jsonl \
#   -c 16

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
from collections import defaultdict

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Ref: https://github.com/haoheliu/audioldm_eval/tree/main
# This script uses a locally modified version of audioldm_eval.
from audioldm_eval import EvaluationHelper

# Ref: https://github.com/LAION-AI/CLAP
# The ref command for installing: pip install laion-clap
import laion_clap

from utils.general import read_jsonl_to_mapping, audio_dir_to_mapping

import os
import shutil
from pathlib import Path


def create_symlink_folder(gen_folder_path: str) -> str:
    gen_folder = Path(gen_folder_path).resolve()
    parent_dir = gen_folder.parent
    link_folder = parent_dir / (gen_folder.name + "_link")

    # 如果软链接目录已存在，先删除
    if link_folder.exists():
        shutil.rmtree(link_folder)

    link_folder.mkdir()

    # 遍历原始目录中的所有文件，创建软链接
    for file in gen_folder.iterdir():
        if file.is_file():
            link_name = link_folder / (file.stem[:11] + '.wav')  # 可自定义重命名逻辑
            link_name.symlink_to(file.resolve())

    return str(link_folder)


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
    ref_aid_to_audios = read_jsonl_to_mapping(
        args.ref_audio_jsonl,
        "audio_id",
        "audio",
        base_path='/cpfs_shared/jiahao.mei/data/tta'
    )
    if args.gen_audio_jsonl is not None:
        gen_aid_to_audios = read_jsonl_to_mapping(
            args.gen_audio_jsonl, "audio_id", "audio"
        )
    elif args.gen_audio_dir is not None:
        gen_aid_to_audios = audio_dir_to_mapping(args.gen_audio_dir, 'tta')

    keys = deepcopy(list(ref_aid_to_audios.keys()))
    for key in keys:
        if key not in gen_aid_to_audios:
            ref_aid_to_audios.pop(key)
    """Calculate ldm eval score: FAD, FD, KL score"""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = "cnn14" if args.task == "tta" else "mert"
    evaluator = EvaluationHelper(16000, args.device, backbone=backbone)
    gen_folder_path, gen_is_same_folder = get_common_folder_path(
        gen_aid_to_audios
    )
    ref_folder_path, ref_is_same_folder = get_common_folder_path(
        ref_aid_to_audios
    )
    assert gen_is_same_folder == True, "Generated audio files must be in the same folder."
    assert ref_is_same_folder == True, "Reference audio files must be in the same folder."
    gen_folder_path_symlink = create_symlink_folder(gen_folder_path)
    eval_result = evaluator.main(
        gen_folder_path_symlink,
        ref_folder_path,
        recalculate=False,
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
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn
    )
    clap_scorer = laion_clap.CLAP_Module(enable_fusion=False)
    # If CLAP fails to load, set verbose=False to True to check errors.
    clap_model_path = os.environ["CLAP_MODEL_PATH"]
    assert clap_model_path is not None, "CLAP_MODEL_PATH environment variable not set."
    clap_scorer.load_ckpt(ckpt=clap_model_path, verbose=False)
    clap_scorer.eval()
    for batch in tqdm(dataloader, desc="Computing CLAP score"):
        scores = compute_clap_metrics(batch, clap_scorer)
        for audio_id, score in zip(batch["audio_id"], scores):
            results["CLAP_score"][audio_id] = score.item()

    with open(args.output_file, "w") as writer:
        for metric, values in results.items():
            if metric == "CLAP_score":
                print_msg = f"{metric}: {np.mean(list(values.values())):.3f}"
                print(print_msg)
                print(print_msg, file=writer)
                if args.clap_per_audio:
                    for audio_id, score in values.items():
                        score_msg = f"{audio_id}: {score[0][0]:.3f}"
                        #print(score_msg)
                        print(score_msg, file=writer)

            else:
                print_msg = f"{metric}: {values:.3f}"
                print(print_msg)
                print(print_msg, file=writer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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
        default="tta",
        help="task type, text-to-audio (tta) or text_to_music (ttm)",
        choices=["tta", "ttm"]
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

    args = parser.parse_args()

    evaluate(args)
