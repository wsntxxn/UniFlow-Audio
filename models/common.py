from typing import Sequence

from accel_hydra.utils.torch import create_mask_from_length, loss_with_mask
import torch
import torch.nn.functional as F

from utils.torch_utilities import create_alignment_path


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
        pred = torch.round(pred * self.latent_token_rate)
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
        result = torch.round(global_pred * self.latent_token_rate)
        # avoid error accumulation for each frame
        if use_local:
            pred_from_local = local_pred.sum(1)
            result[is_time_aligned] = pred_from_local[is_time_aligned]

        result = result.long()
        return result

    def expand_by_duration(
        self,
        x: torch.Tensor,
        content_mask: torch.Tensor,
        local_duration: torch.Tensor,
        global_duration: torch.Tensor,
    ):
        training = getattr(self, 'training', False)
        if not training:  # inference mode
            latent_length = global_duration
        else:  # training mode
            latent_length = local_duration.sum(1)
        latent_mask = create_mask_from_length(latent_length).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)
        align_path = create_alignment_path(local_duration, attn_mask)
        expanded_x = torch.matmul(align_path.transpose(1, 2).to(x.dtype), x)
        return expanded_x, latent_mask
