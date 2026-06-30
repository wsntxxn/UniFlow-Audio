from typing import Any

import torch
import torch.nn as nn
from accel_hydra.utils.torch import concat_non_padding, restore_from_concat, create_mask_from_length

from models.content_encoder.content_encoder import ContentEncoder
from models.dit.modules import SinusoidalPositionalEncoding


######################
# fastspeech modules
######################
class LayerNorm(nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """
    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm,
                     self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        padding: str = "SAME"
    ):
        super(DurationPredictor, self).__init__()
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = in_channels if idx == 0 else filter_channels
            self.conv += [
                nn.Sequential(
                    nn.ConstantPad1d(((kernel_size - 1) // 2,
                                      (kernel_size - 1) //
                                      2) if padding == 'SAME' else
                                     (kernel_size - 1, 0), 0),
                    nn.Conv1d(
                        in_chans,
                        filter_channels,
                        kernel_size,
                        stride=1,
                        padding=0
                    ), nn.ReLU(), LayerNorm(filter_channels, dim=1),
                    nn.Dropout(p_dropout)
                )
            ]
        self.linear = nn.Linear(filter_channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        # x: [B, T, E]
        x = x.transpose(1, -1)
        x_mask = x_mask.unsqueeze(1).to(x.device)
        for f in self.conv:
            x = f(x)
            x = x * x_mask.float()

        x = self.linear(x.transpose(1, -1)
                       ) * x_mask.transpose(1, -1).float()  # [B, T, 1]
        return x


######################
# adapter modules
######################


class ContentAdapterBase(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.d_out = d_out


class CrossAttentionAdapter(ContentAdapterBase):
    def __init__(
        self,
        d_out: int,
        content_dim: int,
        prefix_dim: int,
        num_heads: int,
        duration_predictor: DurationPredictor | None = None,
        dropout: float = 0.1,
        duration_grad_scale: float = 0.1,
    ):
        super().__init__(d_out)
        self.ta_attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=prefix_dim,
            vdim=prefix_dim,
            batch_first=True,
        )
        self.ta_norm = nn.LayerNorm(content_dim)
        self.ta_proj = nn.Conv1d(content_dim, d_out, 1)

        self.nta_attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=prefix_dim,
            vdim=prefix_dim,
            batch_first=True,
        )
        self.nta_norm = nn.LayerNorm(content_dim)
        self.nta_proj = nn.Conv1d(content_dim, d_out, 1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=content_dim,
            vdim=content_dim,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(content_dim)

        self.duration_predictor = duration_predictor
        if duration_predictor is not None:
            self.duration_grad_scale = duration_grad_scale
            self.global_duration_mlp = nn.Sequential(
                nn.Linear(content_dim, content_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(content_dim, 1)
            )

    def forward(
        self,
        ta_content,
        ta_content_mask,
        nta_content,
        nta_content_mask,
        prefix,
        prefix_mask,
    ):
        ta_output, ta_weights = self.ta_attn(
            query=ta_content,
            key=prefix,
            value=prefix,
            key_padding_mask=~prefix_mask.bool()
        )
        ta_output = ta_output * ta_content_mask.unsqueeze(-1).float()
        x = self.ta_norm(ta_output + ta_content)
        ta_content = self.ta_proj(x.transpose(1, 2)).transpose(1, 2)
        nta_output, nta_weights = self.nta_attn(
            query=nta_content,
            key=prefix,
            value=prefix,
            key_padding_mask=~prefix_mask.bool()
        )
        nta_output = nta_output * nta_content_mask.unsqueeze(-1).float()
        x = self.nta_norm(nta_output + nta_content)
        nta_content = self.nta_proj(x.transpose(1, 2)).transpose(1, 2)

        cross_output, cross_weights = self.cross_attn(
            query=ta_content,
            key=nta_content,
            value=nta_content,
            key_padding_mask=~nta_content_mask.bool()
        )
        cross_output = cross_output * ta_content_mask.unsqueeze(-1).float()
        x = self.cross_norm(cross_output + ta_content)
        x_grad_rescaled = x * self.duration_grad_scale + x.detach(
        ) * (1 - self.duration_grad_scale)
        x_aggregated = (
            x_grad_rescaled * ta_content_mask.unsqueeze(-1).float()
        ).sum(dim=1) / ta_content_mask.sum(dim=1, keepdim=True).float()
        global_duration = self.global_duration_mlp(x_aggregated).squeeze(-1)
        local_duration = self.duration_predictor(
            x_grad_rescaled, ta_content_mask
        ).squeeze(-1)
        return ta_content, nta_content, global_duration, local_duration


class ContentEncoderAdapterMixin:
    def __init__(
        self,
        content_encoder: ContentEncoder,
        content_adapter: CrossAttentionAdapter | None = None,
        # None: compatible with single task diffusion/fm
    ):
        self.content_encoder = content_encoder
        self.content_adapter = content_adapter

    def encode_content(
        self,
        content: list[Any],
        task: list[str],
        device: str | torch.device,
        instruction: torch.Tensor | None = None,
        instruction_lengths: torch.Tensor | None = None
    ):

        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        ta_content, ta_content_mask = content_output[
            "ta_content"], content_output["ta_content_mask"]
        nta_content, nta_content_mask = content_output[
            "nta_content"], content_output["nta_content_mask"]

        if instruction is not None:
            instruction_mask = create_mask_from_length(instruction_lengths)
            (
                ta_content,
                nta_content,
                global_duration_pred,
                local_duration_pred,
            ) = self.content_adapter(
                ta_content, ta_content_mask, nta_content, nta_content_mask,
                instruction, instruction_mask
            )

        return_dict = {
            "ta_content": ta_content,
            "ta_content_mask": ta_content_mask,
            "nta_content": nta_content,
            "nta_content_mask": nta_content_mask,
            "la_content": content_output["la_content"],
        }
        if instruction is not None:
            return_dict["global_duration_pred"] = global_duration_pred
            return_dict["local_duration_pred"] = local_duration_pred

        return return_dict
