from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .modules import MLP, ChannelLastConv1d, ConvMLP


def apply_rope(x: torch.Tensor,
               rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(device_type='cuda', enabled=False):
        _x = x.float()
        _x = _x.view(*_x.shape[:-1], -1, 1, 2)
        x_out = rot[..., 0] * _x[..., 0] + rot[..., 1] * _x[..., 1]
        return x_out.reshape(*x.shape).to(dtype=x.dtype)


def compute_rope_rotations(
    length: int,
    dim: int,
    theta: int,
    *,
    freq_scaling: float = 1.0,
    device: torch.device | str = 'cpu'
) -> torch.Tensor:
    assert dim % 2 == 0

    with torch.amp.autocast(device_type='cuda', enabled=False):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        freqs = 1.0 / (
            theta**(
                torch.arange(0, dim, 2, dtype=torch.float32, device=device) /
                dim
            )
        )
        freqs *= freq_scaling

        rot = torch.einsum('..., f -> ... f', pos, freqs)
        rot = torch.stack([
            torch.cos(rot), -torch.sin(rot),
            torch.sin(rot),
            torch.cos(rot)
        ],
                          dim=-1)
        rot = rearrange(rot, 'n d (i j) -> 1 n d i j', i=2, j=2)
        return rot


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
):
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if mask is not None:
        # Convert key padding mask [B, S] to a broadcastable attention mask
        # shape [B, 1, 1, S] for SDPA.
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            # Allow [B, L, S] masks by inserting a head axis.
            mask = mask[:, None, :, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out


class SelfAttention(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange(
            'b n (h d j) -> b h n d j', h=nheads, d=dim // nheads, j=3
        )

    def pre_attention(
        self, x: torch.Tensor, rot: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
        out = attention(q, k, v)
        return out


class MMDitSingleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        pre_only: bool = False,
        kernel_size: int = 7,
        padding: int = 3
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)

        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
            )
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(
                    dim, dim, kernel_size=kernel_size, padding=padding
                )
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(
                    dim,
                    int(dim * mlp_ratio),
                    kernel_size=kernel_size,
                    padding=padding
                )

            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
            )

    def pre_attention(
        self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]
    ):
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(
        self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]
    ):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp

        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rot: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv, mask=mask)
        x = self.post_attention(x, attn_out, x_conditions)

        return x


class JointBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        pre_only: bool = False
    ):
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=False, kernel_size=3, padding=1
        )
        self.nta_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1
        )

    def forward(
        self,
        latent: torch.Tensor,
        nta_content: torch.Tensor,
        nta_content_global: torch.Tensor,
        ta_content: torch.Tensor,
        latent_rot: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
        nta_content_mask: Optional[torch.Tensor] = None,
    ):
        x_qkv, x_mod = self.latent_block.pre_attention(
            latent, ta_content, latent_rot
        )
        nta_qkv, nta_mod = self.nta_block.pre_attention(
            nta_content, nta_content_global, rot=None
        )

        latent_len = latent.shape[1]

        joint_qkv = [
            torch.cat([x_qkv[i], nta_qkv[i]], dim=2) for i in range(3)
        ]

        attn_mask = None
        if latent_mask is not None or nta_content_mask is not None:
            if latent_mask is None:
                latent_mask = torch.ones(
                    latent.shape[0],
                    latent.shape[1],
                    dtype=torch.bool,
                    device=latent.device
                )
            if nta_content_mask is None:
                nta_content_mask = torch.ones(
                    nta_content.shape[0],
                    nta_content.shape[1],
                    dtype=torch.bool,
                    device=nta_content.device
                )
            attn_mask = torch.cat([latent_mask, nta_content_mask], dim=1)

        attn_out = attention(*joint_qkv, mask=attn_mask)
        x_attn_out = attn_out[:, :latent_len]
        nta_attn_out = attn_out[:, latent_len:]

        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
        if not self.pre_only:
            nta_content = self.nta_block.post_attention(
                nta_content, nta_attn_out, nta_mod
            )

        return latent, nta_content


class FinalBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent
