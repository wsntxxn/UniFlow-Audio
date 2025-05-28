from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_utilities import create_mask_from_length


class MlpVideoEncoder(nn.Module):
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

    def forward(self, frames: torch.Tensor, frame_nums: Sequence[int]):
        device = frames.device
        x = F.normalize(frames, p=2, dim=-1)
        x = self.mlp(x)
        mask = create_mask_from_length(frame_nums).to(device)
        return {"output": x, "mask": mask}
