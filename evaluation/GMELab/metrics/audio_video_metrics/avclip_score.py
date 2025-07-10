from typing import List
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from submodules.Synchformer.utils.utils import check_if_file_exists_else_download
from submodules.Synchformer.dataset.dataset_utils import get_video_and_audio
from submodules.Synchformer.scripts.train_utils import (
    get_model,
    get_transforms,
    prepare_inputs,
)
from tqdm import tqdm
from dataclasses import dataclass

from metrics.sync import repeat_video

# this is basically the same as InSyncCfg
@dataclass
class AVClipScoreCfg:
    exp_name: str = "24-01-04T16-39-21"
    device: str = "cuda:0"
    vfps: int = 25
    afps: int = 16_000
    input_size: int = 256
    ckpt_parent_path: str = "./checkpoints/sync_models"

    def __post_init__(self):
        # TODO: checking
        pass


def calculate_avclip_score(
    samples: str,
    exp_name: str,
    afps: int,
    vfps: int,
    input_size: int,
    device: str,
    ckpt_parent_path: str,
    verbose: bool = False,
) -> float:
    cfg_path = f"{ckpt_parent_path}/{exp_name}/cfg-{exp_name}.yaml"
    ckpt_path = f"{ckpt_parent_path}/{exp_name}/{exp_name}.pt"

    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # load config
    model_cfg = OmegaConf.load(cfg_path)
    generated_videos_path = Path(samples)
    model_cfg.model.params.afeat_extractor.params.agg_time_module = "AveragePooling"
    model_cfg.model.params.vfeat_extractor.params.agg_time_module = "AveragePooling"

    if model_cfg.data.vfps != vfps:
        print(
            "WARNING: The model was trained with a different vfps than the provided one"
        )
    if model_cfg.data.afps != afps:
        print(
            "WARNING: The model was trained with a different afps than the provided one"
        )
    if model_cfg.data.size_before_crop != input_size:
        print(
            "WARNING: The model was trained with a different input_size than the provided one"
        )

    device = torch.device(device)

    # load the model
    _, model = get_model(model_cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model"])

    model.eval()
    transforms = get_transforms(model_cfg, ["test"])["test"]

    # init the metric results
    results: List[float] = []

    batch = []
    videos = list(generated_videos_path.glob("*.mp4"))
    original_video_dir = Path(samples).parts[-1]
    assert len(videos), f"No videos found in {samples}... Problems with reencoding?"
    for i, vid_path in tqdm(
        enumerate(videos), desc="Calculating AVClip score", total=len(videos)
    ):
        vid_path_str = vid_path.as_posix()
        try:
            # load visual and audio streams
            # (Tv, 3, H, W) in [0, 255], (Ta, C) in [-1, 1]
            rgb, audio, meta = get_video_and_audio(vid_path_str, get_meta=True)
            rgb, audio = repeat_video(
                rgb, audio, vfps, afps, model_cfg.data.crop_len_sec
            )
            audio = torch.rand_like(audio)  # dummy audio
            item = {
                "video": rgb,
                "audio": audio,
                "meta": meta,
                "path": f"{original_video_dir}/{vid_path.name}",
                "split": "test",
                "targets": {
                    # setting the start of the visual crop and the offset size.
                    # For instance, if the model is trained on 5sec clips, the provided video is 9sec, and `v_start_i_sec=1.3`
                    # the transform will crop out a 5sec-clip from 1.3 to 6.3 seconds and shift the start of the audio
                    # track by `args.offset_sec` seconds. It means that if `offset_sec` > 0, the audio will
                    # start `offset_sec` earlier than the rgb track.
                    # It is a good idea to use something in [-`max_off_sec`, `max_off_sec`] (see `grid`)
                    "v_start_i_sec": 0,
                    "offset_sec": 0,
                    # dummy values -- don't mind them
                    "vggsound_target": 0,
                    "vggsound_label": "PLACEHOLDER",
                },
            }
            # applying the transform
            item = transforms(item)
        except Exception as e:
            print(f"Error while transforming {vid_path_str}: {e}")
            continue
        batch.append(item)
        if len(batch) == 1 or i == len(videos) - 1:
            # prepare inputs for inference
            batch = torch.utils.data.default_collate(batch)
            aud, vid, targets = prepare_inputs(batch, device)

            # forward pass
            with torch.autocast("cuda", enabled=model_cfg.training.use_half_precision):
                with torch.set_grad_enabled(False):
                    vis, aud = model(
                        vid, aud, targets["offset_target"], return_embs=True
                    )
            vis, aud = vis.detach().cpu(), aud.detach().cpu()
            assert (
                vis.shape == aud.shape
            ), "Visual and audio embeddings must have the same shape"
            if vis.dim() == 3:
                B, S, D = vis.shape
                vis = vis.view(B * S, D)
                aud = aud.view(B * S, D)
            # gather similarity scores of corresponding pairs
            results += torch.diag(
                pairwise_cosine_similarity(
                    vis,
                    aud,
                    reduction=None,
                )
            ).tolist()

            batch = []

    score = float(sum(results) / len(results))
    if verbose:
        print("AVClip score:", score)
        print("Result vector:", results)

    return score
