import random
import importlib
from typing import Any
from typing import Sequence

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor

from models.conditional_flow_matching_lib import conditional_flow_matching
from models.content_encoder.content_encoder import ContentEncoder
from models.common import LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase


class FlowMatcherMixin:

    def __init__(
        self, 
        cfm_type: str='default',
        # sigma: Union[float, int] = 0.0, # simply follow the default value in cfm lib
        classifier_free_guidance: bool = True,
        sample_strategy: str='euler'
    ):
        r"""
        Initialize the ConditionalFlowMatcher class.
        Parameters
        -------------
        cfm_type: string index for a specific Conditional Flow Matching class.
        sample_strategy: the method sampling from vector field.        
        """
        self.cfm_map = {
            'default': 'ConditionalFlowMatcher',
            'exact_optimal_transport': 'ExactOptimalTransportConditionalFlowMatcher',
            'target': 'TargetConditionalFlowMatcher',
            'schrodinger_bridge': 'SchrodingerBridgeConditionalFlowMatcher',
            'variance_preserving': 'VariancePreservingConditionalFlowMatcher',
        }
        cfm_name = self.cfm_map.get(cfm_type)
        if cfm_name is None:
            raise NotImplementedError(f"cfm_type '{cfm_type}' is not implemented.")

        self.flow_matcher = getattr(conditional_flow_matching, cfm_name)()
        self.classifier_free_guidance = classifier_free_guidance
        self.sample_strategy = sample_strategy

    def get_target(
        self,
        latent: torch.Tensor,
    ):
        x0 = torch.randn_like(latent)
        x1 = latent
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0=x0, x1=x1)

        return t, xt, ut

    def get_timesteps(
        self,
        num_inference_steps: int=200,
    ):
        if self.sample_strategy == 'euler':
            dt = 1.0 / num_inference_steps
            timesteps = [(1. - i/num_inference_steps) for i in range(num_inference_steps)]
        # NOTE: complete other sampling methods

        return timesteps.long(), dt



class AudioFlowMatching(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    FlowMatcherMixin
):
    def __init__(
        self,
        autoencoder: nn.Module,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        classifier_free_guidance: bool = True,
        sample_strategy: str='euler',
        flow_matching_type: str='default',
    ):
        nn.Module.__init__(self)
        FlowMatcherMixin.__init__(
            self, flow_matching_type, classifier_free_guidance, sample_strategy
        )

        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.content_encoder = content_encoder
        self.backbone = backbone
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self, content: list[Any], condition: list[Any], task: list[str],
        waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ):
        device = self.dummy_param.device

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )
        
        content, content_mask = self.content_encoder.encode_content(
            content, task
        )

        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < 0.1
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        batch_size = latent.shape[0]


        t, xt, ut = self.get_target(latent)

        vt = self.backbone(
            x=xt,
            timesteps=t,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )

        loss = F.mse_loss(vt.float(), ut.float(), reduction="mean")

        return loss

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        latent_shape: Sequence[int],
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        num_samples_per_content: int = 1,
        disable_progress: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(content) * num_samples_per_content

        if classifier_free_guidance:
            content, content_mask = self.encode_content_classifier_free(
                content, task, num_samples_per_content
            )
        else:
            content, content_mask = self.content_encoder.encode_content(
                content, task
            )
            content = content.repeat_interleave(num_samples_per_content, 0)
            content_mask = content_mask.repeat_interleave(
                num_samples_per_content, 0
            )

        latent = self.prepare_latent(batch_size, latent_shape, content.dtype, device)

        timesteps, dt = self.get_timesteps(num_inference_steps=num_steps)
        for i, t in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            latent_input = torch.cat([latent] * 2) if classifier_free_guidance else latent

            t = torch.tensor([t], dtype=latent.dtype, device=latent.device)
            pred = self.backbone(
                x=latent_input, 
                timesteps=t,
                context=content,
                context_mask=content_mask
            )

            # perform guidance
            if classifier_free_guidance:
                pred_uncond, pred_content = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (
                    pred_content - pred_uncond
                )
            latent = latent + pred * dt
            if i == len(timesteps) - 1:
                progress_bar.update(1)
        
        waveform = self.autoencoder.decode(latent)

        return waveform


    def prepare_latent(
        self, batch_size: int,
        latent_shape: Sequence[int], dtype: torch.dtype, device: str
    ):
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=dtype
        )
        # TODO: Check if flow matching requires scaling the latent
        return latent


    def encode_content_classifier_free(
        self,
        content: list[Any],
        task: list[str],
        num_samples_per_content: int = 1
    ):
        content, content_mask = self.content_encoder.encode_content(
            content, task
        )

        content = content.repeat_interleave(num_samples_per_content, 0)
        content_mask = content_mask.repeat_interleave(
            num_samples_per_content, 0
        )

        # get unconditional embeddings for classifier free guidance
        uncond_content = torch.zeros_like(content)
        uncond_content_mask = content_mask.detach().clone()

        uncond_content = uncond_content.repeat_interleave(
            num_samples_per_content, 0
        )
        uncond_content_mask = uncond_content_mask.repeat_interleave(
            num_samples_per_content, 0
        )

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        content = torch.cat([uncond_content, content])
        content_mask = torch.cat([uncond_content_mask, content_mask])

        return content, content_mask
