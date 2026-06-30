import logging

from accel_hydra.utils.torch import create_mask_from_length, contains_nan
import torch

logger = logging.Logger(__file__)


def convert_pad_shape(pad_shape: list[list[int]]):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def create_alignment_path(duration: torch.Tensor, mask: torch.Tensor):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = create_mask_from_length(cum_duration_flat, t_y).float()
    path = path.view(b, t_x, t_y)
    # take the diff on the `t_x` axis
    path = path - torch.nn.functional.pad(
        path, convert_pad_shape([[0, 0], [1, 0], [0, 0]])
    )[:, :-1]
    path = path * mask
    return path


def check_nan_in_batch(batch):
    """check if batch contains NaN and return nan audio ids"""
    assert type(batch) == dict, "batch type error"
    nan_audio_ids = []
    audio_ids = batch["audio_id"]
    audio_id2content = {}
    for idx, audio_id in enumerate(audio_ids):
        content = []
        for k, v in batch.items():
            if k == "audio_id":
                continue
            content.append(v[idx])
        audio_id2content[audio_id] = content

    for audio_id, content in audio_id2content.items():
        if contains_nan(content):
            nan_audio_ids.append(audio_id)
            print(f"{audio_id} contains NaN")
    return nan_audio_ids
