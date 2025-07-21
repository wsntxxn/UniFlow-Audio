from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_utilities import (
    load_pretrained_model, merge_matched_keys, create_mask_from_length,
    loss_with_mask, create_alignment_path
)


class LoadPretrainedBase(nn.Module):
    def process_state_dict(
        self, model_dict: dict[str, torch.Tensor],
        state_dict: dict[str, torch.Tensor]
    ):
        """
        Custom processing functions of each model that transforms `state_dict` loaded from 
        checkpoints to the state that can be used in `load_state_dict`.
        Use `merge_mathced_keys` to update parameters with matched names and shapes by 
        default.  

        Args
            model_dict:
                The state dict of the current model, which is going to load pretrained parameters
            state_dict:
                A dictionary of parameters from a pre-trained model.

            Returns:
                dict[str, torch.Tensor]:
                    The updated state dict, where parameters with matched keys and shape are 
                    updated with values in `state_dict`.      
        """
        state_dict = merge_matched_keys(model_dict, state_dict)
        return state_dict

    def load_pretrained(self, ckpt_path: str | Path):
        load_pretrained_model(
            self, ckpt_path, state_dict_process_fn=self.process_state_dict
        )


class CountParamsBase(nn.Module):
    def count_params(self):
        num_params = 0
        trainable_params = 0
        for param in self.parameters():
            num_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return num_params, trainable_params


class SaveTrainableParamsBase(nn.Module):
    @property
    def param_names_to_save(self):
        names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                names.append(name)
        for name, _ in self.named_buffers():
            names.append(name)
        return names

    def load_state_dict(self, state_dict, strict=True):
        for key in self.param_names_to_save:
            if key not in state_dict:
                raise Exception(
                    f"{key} not found in either pre-trained models (e.g. BERT)"
                    " or resumed checkpoints (e.g. epoch_40/model.pt)"
                )
        return super().load_state_dict(state_dict, strict)


class DurationAdapterMixin:
    def __init__(
        self,
        latent_token_rate: int,
        offset: float = 1.0,
        frame_resolution: float | None = None
    ):
        self.latent_token_rate = latent_token_rate
        self.offset = offset
        self.frame_resolution = frame_resolution

    def get_global_duration_loss(
        self,
        pred: torch.Tensor,
        latent_mask: torch.Tensor,
        reduce: bool = True,
    ):
        target = torch.log(
            latent_mask.sum(1) / self.latent_token_rate + self.offset
        )
        loss = F.mse_loss(target, pred, reduction="mean" if reduce else "none")
        return loss

    def get_local_duration_loss(
        self, ground_truth: torch.Tensor, pred: torch.Tensor,
        mask: torch.Tensor, is_time_aligned: Sequence[bool], reduce: bool
    ):
        n_frames = torch.round(ground_truth / self.frame_resolution)
        target = torch.log(n_frames + self.offset)
        loss = loss_with_mask(
            (target - pred)**2,
            mask,
            reduce=False,
        )
        loss *= is_time_aligned
        if reduce:
            if is_time_aligned.sum().item() == 0:
                loss *= 0.0
                loss = loss.mean()
            else:
                loss = loss.sum() / is_time_aligned.sum()

        return loss

    def prepare_local_duration(self, pred: torch.Tensor, mask: torch.Tensor):
        pred = torch.exp(pred) * mask
        pred = torch.ceil(pred) - self.offset
        pred *= self.frame_resolution
        return pred

    def prepare_global_duration(
        self,
        global_pred: torch.Tensor,
        local_pred: torch.Tensor,
        is_time_aligned: Sequence[bool],
        use_local: bool = True,
    ):
        """
        global_pred: predicted duration value, processed by logarithmic and offset
        local_pred: predicted latent length 
        """
        global_pred = torch.exp(global_pred) - self.offset
        result = global_pred
        # avoid error accumulation for each frame
        if use_local:
            pred_from_local = torch.round(local_pred * self.latent_token_rate)
            pred_from_local = pred_from_local.sum(1) / self.latent_token_rate
            result[is_time_aligned] = pred_from_local[is_time_aligned]

        return result

    def expand_by_duration(
        self,
        x: torch.Tensor,
        content_mask: torch.Tensor,
        local_duration: torch.Tensor,
        global_duration: torch.Tensor | None = None,
    ):
        n_latents = torch.round(local_duration * self.latent_token_rate)
        if global_duration is not None:
            latent_length = torch.round(
                global_duration * self.latent_token_rate
            )
        else:
            latent_length = n_latents.sum(1)
        latent_mask = create_mask_from_length(latent_length).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)
        align_path = create_alignment_path(n_latents, attn_mask)
        expanded_x = torch.matmul(align_path.transpose(1, 2).to(x.dtype), x)
        return expanded_x, latent_mask
