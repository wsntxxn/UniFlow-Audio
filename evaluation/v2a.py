"""

"""

import os
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from accel_hydra.utils.general import read_jsonl_to_mapping

from audioldm_eval import EvaluationHelper
from GMELab.metrics.audio_video_metrics.imagebind_score import calculate_imagebind_score
from GMELab.metrics.audio_video_metrics.sync import calculate_sync, InSyncCfg
from utils.general import audio_dir_to_mapping

torch.multiprocessing.set_sharing_strategy('file_system')
GMELAB_CACHE = os.environ.get(
    "GMELAB_CACHE", os.path.expanduser("~/.cache/gmelab")
)


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
    rank, world_size, device = setup_distributed()
    is_main_process = rank == 0

    results = defaultdict(dict)

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
    else:
        raise ValueError(
            "Either gen_audio_jsonl or gen_audio_dir must be provided."
        )

    keys = deepcopy(list(ref_aid_to_audios.keys()))
    for key in keys:
        if key not in gen_aid_to_audios:
            ref_aid_to_audios.pop(key)

    ref_sync_aid_to_videos = read_jsonl_to_mapping(
        args.sync_ref_video_jsonl, "audio_id", "video"
    )
    """Calculate ImageBind """
    image_bind_score = calculate_imagebind_score(
        aid_to_audio=gen_aid_to_audios,
        device=device,
        vision_embeds=args.ib_vision_embed,
        num_workers=args.num_workers,
    )
    if is_main_process:
        print("image bind: ", image_bind_score)
        results["image_bind_score"] = image_bind_score
    dist_barrier()
    """Calculate Sync """
    overall_sync_score, score_per_video = calculate_sync(
        aid_to_video=ref_sync_aid_to_videos,
        aid_to_audio=gen_aid_to_audios,
        exp_name=InSyncCfg.exp_name,
        afps=InSyncCfg.afps,
        vfps=InSyncCfg.vfps,
        input_size=InSyncCfg.input_size,
        device=device,
        ckpt_parent_path=f"{GMELAB_CACHE}/sync_models",
        num_workers=args.num_workers,
    )
    if is_main_process:
        results["synchformer_score"] = overall_sync_score
        print("sync: ", overall_sync_score)
    """Calculate FAD, FD, KL """
    gen_folder_path, gen_is_same_folder = get_common_folder_path(
        gen_aid_to_audios
    )
    ref_folder_path, ref_is_same_folder = get_common_folder_path(
        ref_aid_to_audios
    )

    assert gen_is_same_folder == True, "Generated audio files must be in the same folder."
    assert ref_is_same_folder == True, "Reference audio files must be in the same folder."

    evaluator = EvaluationHelper(16000, device, backbone="cnn14")

    eval_result = evaluator.main(
        gen_aid_to_audios, ref_aid_to_audios, recalculate=args.recalculate
    )

    assert ref_aid_to_audios.keys() == gen_aid_to_audios.keys(
    ), "Reference and generated audio IDs do not match"

    results.update(eval_result)

    if is_main_process:
        os.makedirs(Path(args.output_file).parent, exist_ok=True)

        with open(args.output_file, "w") as writer:
            for metric, values in results.items():
                print_msg = f"{metric}: {values:.3f}"
                print(print_msg)
                print(print_msg, file=writer)

    dist_barrier()
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_audio_jsonl",
        "-ra",
        type=str,
        required=True,
        help="path to reference audio jsonl file"
    )
    # parser.add_argument(
    #     "--ib_ref_video_jsonl",
    #     "-ibv",
    #     type=str,
    #     required=True,
    #     help="path to reference video jsonl file, original videos"
    # )
    parser.add_argument(
        "--ib_vision_embed",
        "-ibv",
        type=str,
        required=True,
        help="path to reference video ImageBind embedding hdf5 file"
    )
    parser.add_argument(
        "--sync_ref_video_jsonl",
        "-syncv",
        type=str,
        required=True,
        help="path to reference video jsonl file, resampled to 25fps, for "
        "calculating Synchformer score"
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
        default=8,
        type=int,
        help="number of workers for parallel processing"
    )
    parser.add_argument(
        "--recalculate",
        default=False,
        action="store_true",
        help="recalculate metrics if they already exist in the cache"
    )

    args = parser.parse_args()
    evaluate(args)
