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
from models.content_adapter import ContentAdapterBase
from models.common import LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase
from utils.torch_utilities import (
    create_alignment_path, create_mask_from_length, loss_with_mask,
    trim_or_pad_length
)


class FlowMatchingMixin:
    def __init__(
        self,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000
    ) -> None:
        r"""
        Args:
            cfg_drop_ratio (float): Dropout ratio for the autoencoder.
            sample_strategy (str): Sampling strategy for timesteps during training.
            num_train_steps (int): Number of training steps for the noise scheduler.
        """
        self.sample_strategy = sample_strategy
        self.infer_noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_steps
        )
        self.train_noise_scheduler = copy.deepcopy(self.infer_noise_scheduler)

        self.classifier_free_guidance = cfg_drop_ratio > 0.0
        self.cfg_drop_ratio = cfg_drop_ratio

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
            u = torch.randn(bsz, )
        else:
            raise NotImplementedError(
                f"{self.sample_strategy} samlping for timesteps is not supported now"
            )

        indices = (u * self.train_noise_scheduler.config.num_train_timesteps
                  ).long()

        # train_noise_scheduler.timesteps: a list from 1 ~ num_trainsteps with 1 as interval
        timesteps = self.train_noise_scheduler.timesteps[indices].to(
            device=latent.device
        )
        sigmas = self.get_sigmas(
            timesteps, n_dim=latent.ndim, dtype=latent.dtype
        )

        noisy_latent = (1.0 - sigmas) * latent + sigmas * noise

        target = noise - latent

        return noisy_latent, target, timesteps

    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32):
        device = timesteps.device

        # a list from 1 declining to 1/num_train_steps
        sigmas = self.train_noise_scheduler.sigmas.to(
            device=device, dtype=dtype
        )

        schedule_timesteps = self.train_noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [
            (schedule_timesteps == t).nonzero().item() for t in timesteps
        ]

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
        scheduler = self.infer_noise_scheduler

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
            scheduler.set_timesteps(
                timesteps=timesteps, device=device, **kwargs
            )
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
            scheduler.set_timesteps(
                num_inference_steps, device=device, **kwargs
            )
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps


class SingleTaskCrossAttentionAudioFlowMatching(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    FlowMatchingMixin
):
    def __init__(
        self,
        autoencoder: nn.Module,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000,
    ):
        nn.Module.__init__(self)
        FlowMatchingMixin.__init__(
            self, cfg_drop_ratio, sample_strategy, num_train_steps
        )

        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.content_encoder = content_encoder
        self.content_encoder.audio_encoder.model = self.autoencoder
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
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent
        )

        model_pred = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
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
            content_output: dict[
                str, torch.Tensor] = self.content_encoder.encode_content(
                    content, task
                )
            content, content_mask = content_output["content"], content_output[
                "content_mask"]
            content = content.repeat_interleave(num_samples_per_content, 0)
            content_mask = content_mask.repeat_interleave(
                num_samples_per_content, 0
            )

        latent = self.prepare_latent(
            batch_size, latent_shape, content.dtype, device
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = self.retrieve_timesteps(
            num_inference_steps, device, timesteps=None, sigmas=sigmas
        )
        progress_bar = tqdm(
            range(num_inference_steps), disable=disable_progress
        )

        for i, timestep in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            latent_input = torch.cat(
                [latent] * 2
            ) if classifier_free_guidance else latent
            bsz = latent_input.shape[0]

            t = torch.tensor([timestep], device=device)
            t = t.repeat(bsz, )

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

            latent = self.infer_noise_scheduler.step(
                pred, timestep, latent
            ).prev_sample
            if i == len(timesteps) - 1:
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)

        return waveform

    def prepare_latent(
        self, batch_size: int, latent_shape: Sequence[int], dtype: torch.dtype,
        device: str
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


class CrossAttentionAudioFlowMatching(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    FlowMatchingMixin
):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: ContentAdapterBase,
        backbone: nn.Module,
        duration_offset: float = 1.0,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000,
    ):
        nn.Module.__init__(self)
        FlowMatchingMixin.__init__(
            self, cfg_drop_ratio, sample_strategy, num_train_steps
        )
        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.content_encoder = content_encoder
        self.content_encoder.audio_encoder.model = self.autoencoder
        self.content_adapter = content_adapter
        self.backbone = backbone
        self.duration_offset = duration_offset
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self, content: list[Any], task: list[str], waveform: torch.Tensor,
        waveform_lengths: torch.Tensor, instruction: torch.Tensor,
        instruction_lengths: torch.Tensor, **kwargs
    ):
        device = self.dummy_param.device

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        length_aligned_content = content_output["length_aligned_content"]
        content, content_mask = content_output["content"], content_output[
            "content_mask"]
        instruction_mask = create_mask_from_length(instruction_lengths)
        content, content_mask, global_duration_pred, _ = self.content_adapter(
            content, content_mask, instruction, instruction_mask
        )


class DummyContentAudioFlowMatching(CrossAttentionAudioFlowMatching):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: ContentAdapterBase,
        backbone: nn.Module,
        content_dim: int,
        frame_resolution: float,
        duration_offset: float = 1.0,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000
    ):
        super().__init__(
            autoencoder, content_encoder, content_adapter, backbone,
            duration_offset, cfg_drop_ratio, sample_strategy, num_train_steps
        )
