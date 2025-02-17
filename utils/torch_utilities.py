import logging
from typing import Callable
from pathlib import Path
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
    path = create_mask_from_length(cum_duration_flat, t_y).to(mask.dtype)
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
