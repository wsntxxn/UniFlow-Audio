from typing import Sequence

import torch
import torch.nn as nn

from utils.torch_utilities import create_mask_from_length


class MlpVideoEncoder(nn.Module):
    def __init__(
        self,
        video_feat_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(video_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frames: torch.Tensor, frame_nums: Sequence[int]):
        device = frames.device
        x = self.mlp(frames)
        x = self.norm(x)
        mask = create_mask_from_length(frame_nums).to(device)
        return {"output": x, "mask": mask}
