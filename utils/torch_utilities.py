import logging
import math
from typing import Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

logger = logging.Logger(__file__)


def remove_key_prefix_factory(prefix: str = "module."):
    def func(
        model_dict: dict[str, torch.Tensor], state_dict: dict[str,
                                                              torch.Tensor]
    ) -> dict[str, torch.Tensor]:

        state_dict = {
            key[len(prefix):]: value
            for key, value in state_dict.items() if key.startswith(prefix)
        }
        return state_dict

    return func


def merge_matched_keys(
    model_dict: dict[str, torch.Tensor], state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Args:
    model_dict:
        The state dict of the current model, which is going to load pretrained parameters
    state_dict:
        A dictionary of parameters from a pre-trained model.

    Returns:
        dict[str, torch.Tensor]:
            The updated state dict, where parameters with matched keys and shape are 
            updated with values in `state_dict`.
    """
    pretrained_dict = {}
    mismatch_keys = []
    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            pretrained_dict[key] = value
        else:
            mismatch_keys.append(key)
    logger.info(
        f"Loading pre-trained model, with mismatched keys {mismatch_keys}"
    )
    model_dict.update(pretrained_dict)
    return model_dict


def load_pretrained_model(
    model: nn.Module,
    ckpt_or_state_dict: str | Path | dict[str, torch.Tensor],
    state_dict_process_fn: Callable = merge_matched_keys
) -> None:
    state_dict = ckpt_or_state_dict
    if not isinstance(state_dict, dict):
        state_dict = torch.load(ckpt_or_state_dict, "cpu")

    model_dict = model.state_dict()
    state_dict = state_dict_process_fn(model_dict, state_dict)
    model.load_state_dict(state_dict)


def create_mask_from_length(
    lengths: torch.Tensor, max_length: int | None = None
):
    if max_length is None:
        max_length = max(lengths)
    idxs = torch.arange(max_length).reshape(1, -1)  # (1, max_length)
    mask = idxs.to(lengths.device) < lengths.view(-1, 1)
    # (1, max_length) < (batch_size, 1) -> (batch_size, max_length)
    return mask


def loss_with_mask(
    loss: torch.Tensor,
    mask: torch.Tensor,
    reduce: bool = True
) -> torch.Tensor:
    """
    Apply a mask to the loss tensor and optionally reduce it.

    Args:
        loss: Tensor of shape (b, t, ...) representing the loss values.
        mask: Tensor of shape (b, t) where 1 indicates valid positions and 0 indicates masked positions.
        reduce: If True, return a single scalar value; otherwise, return a tensor of shape (b,).

    Returns:
        torch.Tensor: A scalar if reduce is True, otherwise a tensor of shape (b,).
    """
    expanded_mask = mask[(..., ) + (None, ) * (loss.ndim - mask.ndim)]
    expanded_mask = expanded_mask.expand_as(loss)
    masked_loss = loss * expanded_mask

    sum_dims = tuple(range(1, loss.ndim))
    loss_sum = masked_loss.sum(dim=sum_dims)
    mask_sum = expanded_mask.sum(dim=sum_dims)
    loss = loss_sum / mask_sum

    if reduce:
        return loss.mean()
    else:
        return loss


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


def trim_or_pad_length(x: torch.Tensor, target_length: int, length_dim: int):
    """
    Adjusts the size of the specified dimension of tensor x to match `target_length`.
    
    Args:
        x:
            Input tensor.
        target_length: 
            Desired size of the specified dimension.
        length_dim: 
            The dimension to modify.
    
    Returns:
        torch.Tensor: The adjusted tensor.
    """
    current_length = x.shape[length_dim]

    if current_length > target_length:
        # Truncate the tensor
        slices = [slice(None)] * x.ndim
        slices[length_dim] = slice(0, target_length)
        return x[tuple(slices)]

    elif current_length < target_length:
        # Pad the tensor with zeros
        pad_shape = list(x.shape)
        pad_length = target_length - current_length

        pad_shape[length_dim] = pad_length  # Shape for left padding
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)

        return torch.cat([x, padding], dim=length_dim)

    return x


def concat_non_padding(
    seq1: torch.Tensor, mask1: torch.BoolTensor, seq2: torch.Tensor,
    mask2: torch.BoolTensor
):
    """
    Args
        seq1 : Tensor (B, L1, E)
            First sequence.
        mask1 : BoolTensor (B, L1)
            True for valid tokens in seq1, False for padding.
        seq2 : Tensor (B, L2, E)
            Second sequence.
        mask2 : BoolTensor (B, L2)
            True for valid tokens in seq2, False for padding.

    Returns
        concat_seq : Tensor (B, L1+L2, E)
            Both sequences concatenated; valid tokens are left-aligned,
            padding on the right is 0.
        concat_mask: BoolTensor (B, L1+L2)
            Mask for the concatenated sequence.
        perm : LongTensor (B, L1+L2)
            Permutation that maps **original indices → new indices**.
            Needed for restoring the original sequences.
    """
    mask1, mask2 = mask1.bool(), mask2.bool()
    B, L1, E = seq1.shape
    L2 = seq2.size(1)
    L = L1 + L2

    seq_cat = torch.cat([seq1, seq2], dim=1)  # (B, L, E)
    mask_cat = torch.cat([mask1, mask2], dim=1)  # (B, L)

    # ----- Key step: stable sort so that all valid tokens move to the left -----
    # Padding positions get +L, guaranteeing the largest “score” → sorted to the end.
    positions = torch.arange(L, device=seq_cat.device).unsqueeze(0)  # (1, L)
    sort_score = positions + (~mask_cat) * L
    perm = sort_score.argsort(dim=1, stable=True)  # (B, L)

    # Build concatenated sequence & mask
    gather_idx = perm.unsqueeze(-1).expand(-1, -1, E)  # (B, L, E)
    concat_seq = seq_cat.gather(1, gather_idx)
    concat_mask = mask_cat.gather(1, perm)

    # Explicitly zero out the right-hand padding region for safety
    concat_seq = concat_seq * concat_mask.unsqueeze(-1)

    return concat_seq, concat_mask, perm


def restore_from_concat(
    concat_seq: torch.Tensor, mask1: torch.BoolTensor, mask2: torch.BoolTensor,
    perm: torch.LongTensor
):
    """
    Restore (seq1, seq2) from the concatenated sequence produced by
    `concat_non_padding`, using the returned permutation `perm`.
    Fully vectorised — no Python loops.
    """
    mask1, mask2 = mask1.bool(), mask2.bool()
    B, L1 = mask1.shape
    L2 = mask2.size(1)
    E = concat_seq.size(-1)

    # Inverse permutation: maps **new_idx → old_idx**
    inv_perm = torch.empty_like(perm)
    inv_perm.scatter_(
        1, perm,
        torch.arange(L1 + L2, device=perm.device).unsqueeze(0).expand(B, -1)
    )

    # Bring tokens back to their original order
    gather_idx = inv_perm.unsqueeze(-1).expand(-1, -1, E)
    seq_cat_rec = concat_seq.gather(1, gather_idx)  # (B, L1+L2, E)

    # Split back into the two sequences and mask out padding positions
    seq1_restore, seq2_restore = seq_cat_rec.split([L1, L2], dim=1)
    seq1_restore = seq1_restore * mask1.unsqueeze(-1)
    seq2_restore = seq2_restore * mask2.unsqueeze(-1)

    return seq1_restore, seq2_restore


def contains_nan(data):
    """check if data contains NaN"""
    if isinstance(data, torch.Tensor):
        return torch.isnan(data).any().item()
    elif isinstance(data, np.ndarray):
        return np.isnan(data).any()
    elif isinstance(data, float):
        return math.isnan(data)
    elif isinstance(data, (list, tuple)):
        return any(contains_nan(x) for x in data)
    elif isinstance(data, dict):
        return any(contains_nan(v) for v in data.values())
    return False


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
