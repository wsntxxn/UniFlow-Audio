"""
a). Use generative model and generate:
    1. videos with generated audio
    2. generated .wav files
    3. move .wav files to a single folder
    
b). get gen_audio.jsonl and gt_audio.jsonl
    1. run "./generate_postprocess/find_gt_audio.sh" to transfer gt_audio
    2. run "./generate_postprocess/make_video_jsonl.sh" twice for gt_audio and gen_audio

c). Reencode the videos with generated audio on v-fps(25) and a-fps(16000) required by Syncformer.
    1. run "./generate_postprocess/reencode.sh" to resample the gen_video into 16k-Hz
"""

import os
from pathlib import Path
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time

import argparse
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np

from audioldm_eval import EvaluationHelper
from GMELab.metrics.audio_video_metrics.imagebind_score import calculate_imagebind_score
from GMELab.metrics.audio_video_metrics.sync import calculate_sync, InSyncCfg
from utils.general import read_jsonl_to_mapping, audio_dir_to_mapping


def create_symlink_folder(gen_folder_path: str) -> str:
    gen_folder = Path(gen_folder_path).resolve()
    parent_dir = gen_folder.parent
    link_folder = parent_dir / (gen_folder.name + "_link")

    # 如果软链接目录已存在，先删除

    if link_folder.exists():
        shutil.rmtree(link_folder)

    link_folder.mkdir(exist_ok=False)

    # 遍历原始目录中的所有文件，创建软链接
    for file in gen_folder.iterdir():
        if file.is_file():
            link_name = link_folder / (file.stem + '.wav')  # 可自定义重命名逻辑
            link_name.symlink_to(file.resolve())

    return str(link_folder)


def get_common_folder_path(audio_dict):
    """
    
    Params:
        audio_dict: Dict in format {audio_id: audio_path}

    Return:
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
    """ Calculate ImageBind; Sync Scores."""
    # image_bind_score = calculate_imagebind_score(
    #     Path(args.gen_video_path), "cuda"
    # )
    # print("image bind score:", image_bind_score)

    # sync_cfg = InSyncCfg
    # overall_sync_score, score_per_video = calculate_sync(
    #     samples=Path(args.gen_video_path),
    #     exp_name=sync_cfg.exp_name,
    #     afps=sync_cfg.afps,
    #     vfps=sync_cfg.vfps,
    #     input_size=sync_cfg.input_size,
    #     device=sync_cfg.device,
    #     ckpt_parent_path="./evaluation/GMELab/checkpoints/sync_models",
    # )

    # print("sync score:", overall_sync_score)

    # vision_score = {
    #     "image_bind_score": image_bind_score,
    #     "overall_sync_score": overall_sync_score,
    # }

    results = defaultdict(dict)
    # results.update(vision_score)
    """Calculate KL; FAD etc."""
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
        gen_aid_to_audios = audio_dir_to_mapping(args.gen_audio_dir, "v2a")

    keys = deepcopy(list(ref_aid_to_audios.keys()))
    for key in keys:
        if key not in gen_aid_to_audios:
            ref_aid_to_audios.pop(key)
    """Calculate ldm eval score: FAD, FD, KL score"""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_folder_path, gen_is_same_folder = get_common_folder_path(
        gen_aid_to_audios
    )

    ref_folder_path, ref_is_same_folder = get_common_folder_path(
        ref_aid_to_audios
    )

    assert gen_is_same_folder == True, "Generated audio files must be in the same folder."
    assert ref_is_same_folder == True, "Reference audio files must be in the same folder."

    evaluator = EvaluationHelper(16000, args.device, backbone="cnn14")

    eval_result = evaluator.main(
        gen_folder_path, ref_folder_path, recalculate=False
    )

    assert ref_aid_to_audios.keys() == gen_aid_to_audios.keys(
    ), "Reference and generated audio IDs do not match"

    results.update(eval_result)

    os.makedirs(Path(args.output_file).parent, exist_ok=True)

    with open(args.output_file, "w") as writer:
        for metric, values in results.items():
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
        "--gen_video_path",
        type=str,
        # required=True,
        help="path to reencoded generated video with vfps and sr"
    )
    parser.add_argument(
        "--gen_audio_jsonl",
        "-gj",
        type=str,
        help="path to generated audio jsonl file"
    )
    parser.add_argument(
        "--gen_audio_dir",
        "-gd",
        type=str,
        help="path to generated audio directory"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="path to output file"
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        default=4,
        type=int,
        help="number of workers for parallel processing"
    )

    args = parser.parse_args()
    evaluate(args)
