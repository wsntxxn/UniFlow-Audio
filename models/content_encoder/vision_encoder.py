import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel
from einops import rearrange
from typing import Optional

from utils.torch_utilities import create_mask_from_length

class TransformersVideoEncoderBase(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(
        self,
        frames_feature,
    ):  
        vid_len = torch.tensor([frames_feature.shape[0]])
        device = frames_feature.device
        
        frames_feature = self.content_proj(frames_feature)
        frames_feature = self.content_norm(frames_feature)

        mask = create_mask_from_length(vid_len).to(device)
        
        return {
            "output": frames_feature.unsqueeze(0), 
            "mask": mask
        }

class ClipVisionEncoder(TransformersVideoEncoderBase):
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        video_frames_feat_dim: int = 512,
        content_adapter_in_channels: int = 256,
    ):
        nn.Module.__init__(self)
        if model_name != None:
            self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
            self.model = CLIPVisionModel.from_pretrained(model_name)

            for param in self.model.parameters():
                param.requires_grad = False
            self.eval()
        else:
            self.content_proj = nn.Linear(video_frames_feat_dim, content_adapter_in_channels)
            self.content_norm = nn.LayerNorm(content_adapter_in_channels)

    
    def forward(
        self,
        frames_feature,
    ):
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", enabled=False
        ):
            return super().forward(frames_feature)