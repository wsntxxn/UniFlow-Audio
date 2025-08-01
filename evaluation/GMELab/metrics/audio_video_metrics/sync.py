from typing import Dict, Tuple, Union
from pathlib import Path
import os
from math import ceil
from dataclasses import dataclass

from omegaconf import OmegaConf, DictConfig
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from submodules.Synchformer.utils.utils import check_if_file_exists_else_download
from submodules.Synchformer.dataset.dataset_utils import get_video_and_audio
from submodules.Synchformer.scripts.train_utils import (
    get_model,
    get_transforms,
    prepare_inputs,
)
from submodules.Synchformer.dataset.transforms import make_class_grid

GMELAB_CACHE = os.environ.get(
    "GMELAB_CACHE", os.path.expanduser("~/.cache/gmelab")
)


@dataclass
class InSyncCfg:
    exp_name: str = "24-01-04T16-39-21"
    device: str = "cuda:0"
    vfps: int = 25
    afps: int = 16_000
    input_size: int = 256
    ckpt_parent_path: str = "./checkpoints/sync_models"

    def __post_init__(self):
        # TODO: checking
        pass


BATCH_SIZE = 1


def repeat_rgb(
    rgb: torch.Tensor, vfps: float, tgt_len_secs: float
) -> torch.Tensor:
    if tgt_len_secs * vfps > rgb.shape[0]:
        n_repeats = int(tgt_len_secs * vfps / rgb.shape[0]) + 1
        rgb = rgb.repeat(n_repeats, 1, 1, 1)
    rgb = rgb[:ceil(tgt_len_secs * vfps)]
    return rgb


def repeat_audio(
    audio: torch.Tensor, afps: int, tgt_len_secs: float
) -> torch.Tensor:
    if tgt_len_secs * afps > audio.shape[-1]:
        n_repeats = int(tgt_len_secs * afps / audio.shape[-1]) + 1
        # repeat the last dimension
        repeat_pat = [1] * (audio.ndim - 1) + [n_repeats]
        audio = audio.repeat(repeat_pat)
    audio = audio[..., :ceil(tgt_len_secs * afps)]
    return audio


def repeat_video(
    rgb: torch.Tensor, audio: torch.Tensor, vfps: float, afps: int,
    tgt_len_secs: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Repeat the video and audio to match the target length.
    """
    # repeat the video
    rgb = repeat_rgb(rgb, vfps, tgt_len_secs)
    # repeat the audio
    audio = repeat_audio(audio, afps, tgt_len_secs)
    return rgb, audio


# TODO: reorganize the Syncformer and ImageBind import
def modify_model_cfg(model_cfg: DictConfig):
    model_cfg.model.target = "submodules.Synchformer." + model_cfg.model.target
    model_cfg.model.params.afeat_extractor.target = (
        "submodules.Synchformer." +
        model_cfg.model.params.afeat_extractor.target
    )
    model_cfg.model.params.vfeat_extractor.target = (
        "submodules.Synchformer." +
        model_cfg.model.params.vfeat_extractor.target
    )
    model_cfg.model.params.transformer.target = (
        "submodules.Synchformer." + model_cfg.model.params.transformer.target
    )
    model_cfg.model.params.transformer.params.pos_emb_cfg.target = (
        "submodules.Synchformer." +
        model_cfg.model.params.transformer.params.pos_emb_cfg.target
    )
    assert Path(
        f"{GMELAB_CACHE}/sync_models/23-12-22T16-13-38/23-12-22T16-13-38.pt"
    ).exists(
    ), f"The model checkpoint does not exist. Please download the checkpoints using the scripts in {GMELAB_CACHE} folder."
    model_cfg.model.params.afeat_extractor.params.ckpt_path = (
        f"{GMELAB_CACHE}/sync_models/23-12-22T16-13-38/23-12-22T16-13-38.pt"
    )
    model_cfg.model.params.vfeat_extractor.params.ckpt_path = (
        f"{GMELAB_CACHE}/sync_models/23-12-22T16-13-38/23-12-22T16-13-38.pt"
    )
    for t in model_cfg.transform_sequence_train:
        t.target = "submodules.Synchformer." + t.target
    for t in model_cfg.transform_sequence_test:
        t.target = "submodules.Synchformer." + t.target


def infer_sync_single_video(
    audio_path, video_path, afps, vfps, model_cfg, model, transforms, device,
    grid
):
    rgb, audio, meta = get_video_and_audio(
        video_path, audio_path, get_meta=True, afps=afps
    )

    # due to different model sr setting, need extra resample to a_fps
    # or u should reencode the video

    rgb, audio = repeat_video(
        rgb, audio, vfps, afps, model_cfg.data.crop_len_sec
    )
    item = {
        "video": rgb,
        "audio": audio,
        "meta": meta,
        "path": "dummy",
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

    batch = torch.utils.data.default_collate([item])
    aud, vid, targets = prepare_inputs(batch, device)

    # forward pass
    with torch.autocast("cuda", enabled=model_cfg.training.use_half_precision):
        with torch.set_grad_enabled(False):
            _, off_logits = model(vid, aud, targets["offset_target"])
    off_logits = off_logits.detach().cpu()
    off_cls = (
        torch.softmax(off_logits.float(), dim=-1).detach().cpu().argmax(dim=1)
    )
    insync = off_cls == targets["offset_target"].cpu()
    offset_sec = round(grid[off_cls[0].item()].item(), 3)
    return offset_sec


class SyncDataset(Dataset):
    def __init__(
        self,
        video_paths,
        audio_paths,
        vfps,
        afps,
        crop_len_sec,
        transforms,
    ):
        super().__init__()
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.vfps = vfps
        self.afps = afps
        self.crop_len_sec = crop_len_sec
        self.transforms = transforms

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        audio_path = self.audio_paths[idx]

        rgb, audio, meta = get_video_and_audio(
            video_path, audio_path, get_meta=True, afps=self.afps
        )

        rgb, audio = repeat_video(
            rgb, audio, self.vfps, self.afps, self.crop_len_sec
        )

        item = {
            "video": rgb,
            "audio": audio,
            "meta": meta,
            "path": str(video_path),
            "split": "test",
            "targets": {
                "v_start_i_sec": 0.0,
                "offset_sec": 0.0,
                "vggsound_target": 0,
                "vggsound_label": "PLACEHOLDER",
            },
        }
        item = self.transforms(item)
        return item


def calculate_sync(
    aid_to_video: dict[str, str | Path],
    aid_to_audio: dict[str, str | Path],
    exp_name: str,
    afps: int,
    vfps: int,
    input_size: int,
    device: str,
    ckpt_parent_path: str,
    verbose: bool = False,
    num_workers: int = 8,
) -> Tuple[float, Dict[str, Dict[str, Union[int, float, None]]]]:
    cfg_path = f"{ckpt_parent_path}/{exp_name}/cfg-{exp_name}.yaml"
    ckpt_path = f"{ckpt_parent_path}/{exp_name}/{exp_name}.pt"

    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)
    check_if_file_exists_else_download(
        f"{ckpt_parent_path}/23-12-22T16-13-38/23-12-22T16-13-38.pt"
    )

    # load config
    model_cfg = OmegaConf.load(cfg_path)
    modify_model_cfg(model_cfg)

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
    ckpt = torch.load(
        ckpt_path, map_location=torch.device("cpu"), weights_only=False
    )
    model.load_state_dict(ckpt["model"])

    model.eval()
    transforms = get_transforms(model_cfg, ["test"])["test"]

    max_off_sec = model_cfg.data.max_off_sec
    num_cls = model_cfg.data.num_off_cls
    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)

    results: Dict[str, Dict[str, Union[int, float, None]]] = {}
    batch = []

    # get videos
    if aid_to_audio is None:
        all_videos = aid_to_video.values()
        all_audios = all_videos
    else:
        all_videos, all_audios = [], []
        for aid, video in aid_to_video.items():
            if aid in aid_to_audio:
                all_videos.append(video)
                all_audios.append(aid_to_audio[aid])
            else:
                print(f"Audio for {aid} not found, skipping.")

    insync_offsets = 0
    original_video_dir = Path(all_videos[0]).parts[-2]
    assert len(
        all_videos
    ), f"No videos found in {original_video_dir}... Problems with reencoding?"

    dataset = SyncDataset(
        video_paths=all_videos,
        audio_paths=all_audios,
        vfps=vfps,
        afps=afps,
        crop_len_sec=float(model_cfg.data.crop_len_sec),
        transforms=transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=False,
    )

    results = {}
    insync_offsets = 0.0

    for batch in tqdm(loader, desc="Calculating InSync"):
        aud, vid, targets = prepare_inputs(batch, device)
        # forward pass
        with torch.autocast(
            "cuda", enabled=model_cfg.training.use_half_precision
        ), torch.set_grad_enabled(False):
            _, off_logits = model(vid, aud, targets["offset_target"])
        off_logits = off_logits.detach().cpu()
        off_cls = (
            torch.softmax(off_logits.float(),
                          dim=-1).detach().cpu().argmax(dim=1)
        )
        insync = off_cls == targets["offset_target"].cpu()

        for i, path in enumerate(batch["path"]):
            offset_sec = round(grid[off_cls[i].item()].item(), 3)
            insync_offsets += abs(offset_sec)
            results[path] = {
                "insync": insync[i].item(),
                "offset_sec": offset_sec,
                "prob": None,
            }

    score = float(insync_offsets / len(results))
    if verbose:
        print("InSync:", score)
    return score, results


if __name__ == '__main__':
    score, score_per_video = calculate_sync(
        samples=
        "/hpc_stor03/sjtu_home/yaoyun.zhang/project/x_to_audio_generation/evaluation/fad-test/samples/gen-video-5.12s-25fps-16000hz",
        exp_name="24-01-04T16-39-21",
        afps=16000,
        vfps=25,
        input_size=256,
        device="cuda",
        ckpt_parent_path=
        "/hpc_stor03/sjtu_home/yaoyun.zhang/project/x_to_audio_generation/evaluation/GMELab/checkpoints/sync_models",
    )

    print(score)
    # print(score_per_video)
    """
    {'gt-video/1627-train wheels squealing-test.mp4': {'insync': False, 'offset_sec': 0.8, 'prob': None}}
    """
