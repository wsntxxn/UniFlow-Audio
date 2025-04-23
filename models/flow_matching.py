from typing import Any, Optional, Union, List, Sequence

import inspect
import random

from tqdm import tqdm
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from models.autoencoder.autoencoder_base import AutoEncoderBase
from models.content_encoder.content_encoder import ContentEncoder
from models.common import LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase
from utils.torch_utilities import (
    create_alignment_path, create_mask_from_length, loss_with_mask,
    trim_or_pad_length
)


class FlowMatchingMixin:

    def __init__(
        self, 
        
        classifier_free_guidance: bool = True,
        sample_strategy: str='normal',
        num_train_steps: int=1000
    ):
        r"""
        sample_strategy: sample timesteps       
        """
        self.sample_strategy = sample_strategy
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_steps)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.classifier_free_guidance = classifier_free_guidance

    def get_input_target_and_timesteps(
        self,
        latent: torch.Tensor,
    ):  
        bsz = latent.shape[0]
        noise = torch.randn_like(latent)
        
        if self.sample_strategy == 'normal':
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bsz,
                logit_mean=0,
                logit_std=1,
                mode_scale=None,
            )
        elif self.sample_strategy == 'uniform':
            u = torch.randn(bsz,)
        else:
            raise NotImplementedError(f"{self.sample_strategy} samlping for timesteps is not supported now")

        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()

        # noise_scheduler_copy.timesteps: a list from 1 ~ num_trainsteps with 1 as interval 
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latent.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latent.ndim, dtype=latent.dtype)

        noisy_model_input = (1.0 - sigmas) * latent + sigmas * noise

        target = noise - latent

        return noisy_model_input, target, timesteps
    
    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32):
        device = timesteps.device
        
        # a list from 1 declining to 1/num_train_steps 
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)

        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def retrieve_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):  
        # used in inference, retrieve new timesteps on given inference timesteps 
        scheduler = self.noise_scheduler

        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps



class AudioFlowMatching(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    FlowMatchingMixin
):
    def __init__(
        self,
        autoencoder: nn.Module,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        classifier_free_guidance: bool = True,
        sample_strategy: str='normal',
        num_train_steps: int=1000,
    ):
        nn.Module.__init__(self)
        FlowMatchingMixin.__init__(
            self, classifier_free_guidance, sample_strategy, num_train_steps
        )

        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.content_encoder = content_encoder
        self.backbone = backbone
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self, content: list[Any], condition: list[Any], task: list[str],
        waveform: torch.Tensor, waveform_lengths: torch.Tensor, **kwargs
    ):
        device = self.dummy_param.device

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )
        
        content, content_mask = self.content_encoder.encode_content(
            content, task, device=device
        )

        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < 0.1
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        batch_size = latent.shape[0]

        noisy_input, target, timesteps = self.get_input_target_and_timesteps(latent)   

        model_pred = self.backbone(
            x=noisy_input,
            timesteps=timesteps/1000,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        latent_shape: Sequence[int],
        num_inference_steps: int = 50,
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

        latent = self.prepare_latent(batch_size, latent_shape, content.dtype, device)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = self.retrieve_timesteps(
            num_inference_steps, device, timesteps, sigmas
        )
        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)

        for i, orig_t in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            latent_input = torch.cat([latent] * 2) if classifier_free_guidance else latent
            bsz = latent_input.shape[0]

            t = torch.tensor([orig_t / 1000], device=device)
            t = t.repeat(bsz,)

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

            latent = self.noise_scheduler.step(pred, orig_t, latent).prev_sample
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
        return latent


    def encode_content_classifier_free(
        self,
        content: list[Any],
        task: list[str],
        device,
        num_samples_per_content: int = 1
    ):
        content, content_mask = self.content_encoder.encode_content(
            content, task, device=device
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
    
    
class VariableLengthAudioFlowdMatching(AudioFlowMatching):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: nn.Module,
        backbone: nn.Module,
        frame_resolution: float,
        classifier_free_guidance: bool = True,
        sample_strategy: str='normal',
        num_train_steps: int=1000,
        use_content_adapter_feature: bool=False,
    ):
        super().__init__(
            autoencoder, content_encoder, backbone,
            classifier_free_guidance, sample_strategy, num_train_steps
        )
        self.content_adapter = content_adapter
        self.frame_resolution = frame_resolution
        self.dummy_nta_embed = nn.Parameter(torch.zeros(content_adapter.d_out))

        self.dummy_time_align_content = nn.Parameter(torch.zeros(content_adapter.d_out))

        self.use_content_adapter_feature = use_content_adapter_feature

    def forward(
        self, content, condition, duration, duration_lengths, task, waveform,
        waveform_lengths, **kwargs
    ):
        device = self.dummy_param.device

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        # content: (B, L, E)
        content, content_mask = self.content_encoder.encode_content(
            content, task, device=device
        )
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,
        ) = self.content_adapter(content, content_mask)

        # duration = torch.round(duration * self.autoencoder.latent_token_rate)
        n_frames = torch.round(duration / self.frame_resolution)
        local_duration_target = torch.log(n_frames + 1e-8)
        global_duration_target = torch.log(
            latent_mask.sum(1) / self.autoencoder.latent_token_rate + 1e-8
        )
        local_duration_loss = loss_with_mask(
            (local_duration_target - local_duration_pred)**2, content_mask
        )
        global_duration_loss = torch.sum(
            (global_duration_target - global_duration_pred)**2
        ) / content.shape[0]

        # --------------------------------------------------------------------
        # prepare latent and diffusion-related noise
        # --------------------------------------------------------------------
        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < 0.1
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        batch_size = latent.shape[0]
        noisy_input, target, timesteps = self.get_input_target_and_timesteps(latent)

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------

        # content_mask: [B, L], helper_latent_mask: [B, T]
        n_latents = torch.round(duration * self.autoencoder.latent_token_rate)
        helper_latent_mask = create_mask_from_length(n_latents.sum(1)).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1
                                          ) * helper_latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        align_path = create_alignment_path(n_latents, attn_mask)
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2), content
        )  # (B, T, L) x (B, L, E) -> (B, T, E)

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        # TODO compatility for 2D spectrogram VAE
        latent_length = noisy_input.size(self.autoencoder.time_dim)
        if time_aligned_content.size(1) > latent_length:
            time_aligned_content = time_aligned_content[:, :latent_length]
        elif time_aligned_content.size(1) < latent_length:
            pad_size = latent_length - time_aligned_content.size(1)
            padding = (0, 0, 0,
                       pad_size) + (0, 0) * (time_aligned_content.ndim - 2)
            time_aligned_content = F.pad(
                time_aligned_content, padding, mode="constant", value=0
            )

        context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                       (..., )].expand_as(content)
        context_mask = torch.ones(batch_size, context.size(1)).to(device)
        pred: torch.Tensor = self.backbone(
            x=noisy_input,
            timesteps=timesteps / 1000,
            time_aligned_context=time_aligned_content,
            context=context,
            x_mask=latent_mask,
            context_mask=context_mask
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)

        flow_matching_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        
        return {
            "flow_matching_loss": flow_matching_loss,
            "local_duration_loss": local_duration_loss
        }
    
    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        use_gt_duration: bool = False,
        **kwargs
    ):
        device = self.dummy_param.device

        content, content_mask = self.content_encoder.encode_content(
            content, task, device=device
        )
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,  # log(n_frames)
        ) = self.content_adapter(content, content_mask)

        local_duration_pred = torch.exp(local_duration_pred) * content_mask
        local_duration_pred = torch.ceil(local_duration_pred)  # n_frames
        local_duration_pred = torch.round(local_duration_pred * self.frame_resolution * \
            self.autoencoder.latent_token_rate)

        # use ground truth duration
        if use_gt_duration and "duration" in kwargs:
            local_duration_pred = torch.round(
                torch.as_tensor(kwargs["duration"]) *
                self.autoencoder.latent_token_rate
            ).to(device)

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        # content_mask: [B, L], latent_mask: [B, T]
        global_duration = local_duration_pred.sum(1)
        latent_mask = create_mask_from_length(global_duration).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        align_path = create_alignment_path(local_duration_pred, attn_mask)
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2), content
        )  # (B, T, L) x (B, L, E) -> (B, T, E)

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        batch_size = content.size(0)
        latent_shape = tuple(
            int(global_duration.max().item()) if dim is None else dim
            for dim in self.autoencoder.latent_shape
        )
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=content.dtype
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = self.retrieve_timesteps(
            num_inference_steps, device, None, sigmas
        )

        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)

        for i, orig_t in enumerate(timesteps):
            context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                           (..., )].expand_as(content)
            context_mask = torch.ones(batch_size, context.size(1)).to(device)

            t = torch.tensor([orig_t / 1000], device=device)
            t = torch.repeat(batch_size,)

            pred = self.backbone(
                x=latent,
                x_mask=latent_mask,
                timesteps=t,
                time_aligned_context=time_aligned_content,
                context=context,
                context_mask=context_mask
            )

            latent = self.noise_scheduler.step(pred, orig_t, latent).prev_sample

            if i == len(timesteps) - 1:
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)

        return waveform
    
class SameLengthAudioFlowdMatching(VariableLengthAudioFlowdMatching):
    def forward(
        self, content, condition, duration, duration_lengths, task, waveform,
        waveform_lengths, **kwargs
    ):
        device = self.dummy_param.device

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )
        
        # test on text-to-audio task, use dummy time_align_content
        content_input = content
        # content: (B, L, E)
        content, content_mask = self.content_encoder.encode_content(
            content_input, task, device=device
        )
        
        content_feature_copy = content.detach().clone()
        content_mask_copy = content_mask.detach().clone()
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,
        ) = self.content_adapter(content, content_mask)

        # simply treat the whole batch as one single task
        if task[0] == "text_to_audio":
            if self.use_content_adapter_feature:
                context, context_mask = content, content_mask
            else:
                context, context_mask = content_feature_copy,  content_mask_copy

        # duration = torch.round(duration * self.autoencoder.latent_token_rate)
        n_frames = torch.round(duration / self.frame_resolution)
        # TODO handle `n_frames` = 0 scenario here
        local_duration_target = torch.log(n_frames + 1e-8)
        global_duration_target = torch.log(
            latent_mask.sum(1) / self.autoencoder.latent_token_rate + 1e-8
        )
        local_duration_loss = loss_with_mask(
            (local_duration_target - local_duration_pred)**2, content_mask
        )
        global_duration_loss = torch.sum(
            (global_duration_target - global_duration_pred)**2
        ) / content.shape[0]

        # --------------------------------------------------------------------
        # prepare latent and flow-matching target
        # --------------------------------------------------------------------
        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < 0.1
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0
                if task[0] == "text_to_audio":
                    context[mask_indices] = 0 

        batch_size = latent.shape[0]
        noisy_input, target, timesteps = self.get_input_target_and_timesteps(latent)
        
        if task[0] != "text_to_audio":
            # --------------------------------------------------------------------
            # duration adapter
            # --------------------------------------------------------------------
            # content_mask: [B, L], helper_latent_mask: [B, T]
            n_latents = torch.round(duration * self.autoencoder.latent_token_rate)
            helper_latent_mask = create_mask_from_length(n_latents.sum(1)).to(
                content_mask.device
            )
            attn_mask = content_mask.unsqueeze(-1
                                            ) * helper_latent_mask.unsqueeze(1)
            # attn_mask: [B, L, T]
            align_path = create_alignment_path(n_latents, attn_mask)
            time_aligned_content1 = torch.matmul(
                align_path.transpose(1, 2), content
            )  # (B, T, L) x (B, L, E) -> (B, T, E)

            time_aligned_content2 = self.content_encoder.encode_time_aligned_content(
                content_input, task, device=device
            )
            # --------------------------------------------------------------------
            # prepare input to the backbone
            # --------------------------------------------------------------------
            # TODO compatility for 2D spectrogram VAE
            latent_length = noisy_input.size(self.autoencoder.time_dim)
            time_aligned_content1 = trim_or_pad_length(
                time_aligned_content1, latent_length, 1
            )
            time_aligned_content2 = trim_or_pad_length(
                time_aligned_content2, latent_length, 1
            )
            # time_aligned_content1: from unaligned input (phoneme)
            # time_aligned_content2: from aligned input (f0/energy)
            time_aligned_content = time_aligned_content1 + time_aligned_content2
        else:
            latent_length = noisy_input.size(self.autoencoder.time_dim)
            time_aligned_content = self.dummy_time_align_content.expand(batch_size, latent_length, -1).to(device)

        if task[0] != "text_to_audio":
            context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                        (..., )].expand_as(content)
            context_mask = torch.ones(batch_size, context.size(1)).to(device)            

        pred: torch.Tensor = self.backbone(
            x=noisy_input,
            timesteps=timesteps/1000,
            time_aligned_context=time_aligned_content,
            context=context,
            x_mask=latent_mask,
            context_mask=context_mask
        )

        flow_matching_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        
        if task[0] != "text_to_audio": 
            return {
                "flow_matching_loss": flow_matching_loss,
                "local_duration_loss": local_duration_loss
            }
        else:
            return {
                "flow_matching_loss": flow_matching_loss,
                "global_duration_loss": global_duration_loss
            }
    
    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0

        content_input = content
        if classifier_free_guidance:
            content, content_mask = self.encode_content_classifier_free(
                content_input, task, device=device
            )
        else:
            content, content_mask = self.content_encoder.encode_content(
                content_input, task, device=device
            )
        content_feature_copy = content.clone()
        content_mask_copy = content_mask.clone()

        bsz = content.shape[0]

        if task[0] == "text_to_audio":
            context, context_mask = content, content_mask
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,  # log(n_frames)
        ) = self.content_adapter(content, content_mask)

        if task[0] == "text_to_audio":
            if self.use_content_adapter_feature:
                context, context_mask = content, content_mask
            else:
                context, context_mask = content_feature_copy,  content_mask_copy
        
        local_duration = torch.round(
                torch.stack(kwargs["duration"]) * 
                self.autoencoder.latent_token_rate
            ).repeat(bsz, 1).to(device)
        
        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        # content_mask: [B, L], latent_mask: [B, T]
        global_duration = local_duration.sum(1)
        latent_mask = create_mask_from_length(global_duration).to(
            content_mask.device
        )
        if task[0] != "text_to_audio":
            attn_mask = content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)

            # attn_mask: [B, L, T]
            align_path = create_alignment_path(local_duration, attn_mask)
            time_aligned_content1 = torch.matmul(
                align_path.transpose(1, 2), content
            )  # (B, T, L) x (B, L, E) -> (B, T, E)
        
            time_aligned_content2 = self.content_encoder.encode_time_aligned_content(
                content_input, task, device=device
            )
            time_aligned_content2 = trim_or_pad_length(
                time_aligned_content2, time_aligned_content1.size(1), 1
                )
            time_aligned_content = time_aligned_content1 + time_aligned_content2
        else:
            seq_len = latent_mask.shape[1]
            time_aligned_content = self.dummy_time_align_content.expand(bsz, seq_len, -1).to(device)

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        batch_size = len(task)
        latent_shape = tuple(
            int(global_duration.max().item()) if dim is None else dim
            for dim in self.autoencoder.latent_shape
        )
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=content.dtype
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = self.retrieve_timesteps(
            num_inference_steps, device, None, sigmas
        )
        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)

        for i, orig_t in enumerate(timesteps):
            t = torch.tensor([orig_t / 1000], device=device)

            #TODO: enable cfg for other tasks
            if classifier_free_guidance:
                # for tta / v2a only stack the latent and timestep
                # since content is processed in encode-cfg content-encoder
                latent_input = torch.cat([latent] * 2)
                batch_size = latent_input.shape[0]        
                t = t.repeat(batch_size,)
                
                # v2a: content is already cfg-processed, no need stack up context here
                if task[0] !="text_to_audio":
                    context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                                (..., )].expand_as(content)
                    context_mask = torch.ones(batch_size, context.size(1)).to(device)
            else:
                latent_input = latent

            pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=t,
                time_aligned_context=time_aligned_content,
                context=context,
                context_mask=context_mask
            )
            
            if classifier_free_guidance:
                pred_uncond, pred_content = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (
                    pred_content - pred_uncond
                )

            latent = self.noise_scheduler.step(pred, orig_t, latent).prev_sample
            if i == len(timesteps) - 1:
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)
        return waveform
    
