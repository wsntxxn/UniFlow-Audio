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
