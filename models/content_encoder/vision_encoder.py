import random
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from accel_hydra.utils.torch import create_mask_from_length
from x_transformers.x_transformers import Encoder

from ..mmdit.modules import ChannelLastConv1d, ConvMLP, MLP


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        video_feat_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        self.mlp = nn.Linear(video_feat_dim, embed_dim)
        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)

        self.apply(_init_weights)

    def forward(
        self, clip: torch.Tensor, clip_lengths: Sequence[int], **kwargs
    ):
        device = clip.device
        x = F.normalize(clip, p=2, dim=-1)
        x = self.mlp(x)
        mask = create_mask_from_length(clip_lengths).to(device)
        return {
            "ta_output": x,
            "ta_mask": mask,
            "nta_output": None,
            "nta_mask": None
        }


class CLIPInterpolateVideoEncoder(nn.Module):
    def __init__(
        self,
        video_feat_dim: int,
        embed_dim: int,
        latent_token_rate: int,
        transformer_cfg: dict | None = None,
    ):
        super().__init__()
        self.mlp = nn.Linear(video_feat_dim, embed_dim)
        self.latent_token_rate = latent_token_rate
        self.transformer_enhancer = None
        if transformer_cfg is not None:
            self.transformer_enhancer = Encoder(
                dim=embed_dim,
                **transformer_cfg,
            )
        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)

        self.apply(_init_weights)

    def forward(self, clip: torch.Tensor, duration: Sequence[float], **kwargs):
        device = clip.device
        x = F.normalize(clip, p=2, dim=-1)
        x = self.mlp(x)

        target_len = int(max(duration) * self.latent_token_rate)

        x = x.transpose(1, 2)
        x = F.interpolate(x, target_len, mode='nearest-exact')
        x = x.transpose(1, 2)

        latent_length = torch.as_tensor(duration) * self.latent_token_rate
        mask = create_mask_from_length(latent_length).to(device)
        if self.transformer_enhancer is not None:
            x = self.transformer_enhancer(x, mask=mask)

        return {
            "ta_output": x,
            "ta_mask": mask,
            "nta_output": None,
            "nta_mask": None
        }


class CLIPSyncEncoder(nn.Module):
    def __init__(
        self,
        clip_dim: int,
        sync_dim: int,
        embed_dim: int,
        latent_token_rate: int,
        fusion: Literal["add", "concat"] = "add",
        sync_drop_ratio: float = 0.0,
    ) -> None:
        super().__init__()

        self.fusion = fusion
        self.sync_drop_ratio = sync_drop_ratio

        self.clip_proj = nn.Linear(clip_dim, embed_dim)
        self.sync_proj = nn.Linear(sync_dim, embed_dim)
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))
        if fusion == "concat":
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.sync_frame_rate = 25
        self.sync_num_frames_per_segment = 16
        self.sync_step_size = 8
        self.sync_downsample_rate = 2
        self.latent_token_rate = latent_token_rate

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def get_sync_seq_len(self, duration):
        num_frames = duration * self.sync_frame_rate
        num_segments = (
            num_frames - self.sync_num_frames_per_segment
        ) // self.sync_step_size + 1
        sync_seq_len = int(
            num_segments * self.sync_num_frames_per_segment /
            self.sync_downsample_rate
        )
        return sync_seq_len

    def forward(
        self, clip: torch.Tensor, sync: torch.Tensor,
        duration: Sequence[float], clip_lengths: Sequence[int],
        sync_lengths: Sequence[int], **kwargs
    ):

        clip = F.normalize(clip, p=2, dim=-1)
        sync = F.normalize(sync, p=2, dim=-1)

        latent_length = (torch.as_tensor(duration) *
                         self.latent_token_rate).int()
        ta_mask = create_mask_from_length(latent_length).to(sync.device)

        bs = clip.shape[0]
        # target_len = int(max(duration) * self.latent_token_rate)
        target_len = latent_length.max()
        sync_seq_len = self.get_sync_seq_len(max(duration))
        # B * num_segments (24) * 8 * 768
        num_sync_segments = sync_seq_len // 8
        sync = sync.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync = sync.flatten(1, 2)  # (B, VN, D)

        # extend vf to match x
        clip = self.clip_proj(clip)
        sync = self.sync_proj(sync)  # (B, D, VN)

        # upsample the sync features to match the audio
        sync = sync.transpose(1, 2)  # (B, D, VN)
        sync = F.interpolate(sync, size=target_len, mode='nearest-exact')
        sync = sync.transpose(1, 2)  # (B, N, D)

        if self.training:
            if self.sync_drop_ratio > 0:
                if random.random() < self.sync_drop_ratio:
                    sync = torch.zeros_like(sync)

        # upsample the clip features to match the audio
        clip = clip.transpose(1, 2)  # (B, D, VN)
        clip = F.interpolate(clip, size=target_len, mode='nearest-exact')
        clip = clip.transpose(1, 2)  # (B, N, D)

        if self.fusion == "add":
            ta_output = clip + sync
        elif self.fusion == "concat":
            ta_output = self.fusion_proj(torch.cat([clip, sync], dim=-1))
        else:
            raise ValueError(f"Invalid fusion method: {self.fusion}")

        return {
            "ta_output": ta_output,
            "ta_mask": ta_mask,
            "nta_output": None,
            "nta_mask": None
        }
