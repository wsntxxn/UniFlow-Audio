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

        content_output = self.content_encoder.encode_content(
            content, task, device=device
        )
        content, content_mask = content_output["content"], content_output[
            "content_mask"]

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

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss_with_mask(loss, latent_mask)

        return loss

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        latent_shape: Sequence[int],
        num_steps: int = 50,
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

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, num_steps = self.retrieve_timesteps(
            num_steps, device, timesteps=None, sigmas=sigmas
        )
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

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

            progress_bar.update(1)

        progress_bar.close()

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
            autoencoder=autoencoder,
            content_encoder=content_encoder,
            content_adapter=content_adapter,
            backbone=backbone,
            duration_offset=duration_offset,
            cfg_drop_ratio=cfg_drop_ratio,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps
        )
        self.frame_resolution = frame_resolution
        self.dummy_nta_embed = nn.Parameter(torch.zeros(content_dim))
        self.dummy_ta_embed = nn.Parameter(torch.zeros(content_dim))

    def forward(
        self, content: list[Any], duration: Sequence[float], task: list[str],
        is_time_aligned: Sequence[bool], waveform: torch.Tensor,
        waveform_lengths: torch.Tensor, instruction: torch.Tensor,
        instruction_lengths: torch.Tensor, **kwargs
    ):
        device = self.dummy_param.device

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )
        batch_size = latent.size(0)

        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        length_aligned_content = content_output["length_aligned_content"]
        content, content_mask = content_output["content"], content_output[
            "content_mask"]
        context_mask = content_mask.detach()
        instruction_mask = create_mask_from_length(instruction_lengths)

        content, content_mask, global_duration_pred, local_duration_pred = \
            self.content_adapter(content, content_mask, instruction, instruction_mask)

        n_frames = torch.round(duration / self.frame_resolution)
        local_duration_target = torch.log(n_frames + self.duration_offset)
        global_duration_target = torch.log(
            latent_mask.sum(1) / self.autoencoder.latent_token_rate +
            self.duration_offset
        )

        # truncate unused non time aligned duration prediction
        if is_time_aligned.sum() > 0:
            trunc_ta_length = content_mask[is_time_aligned].sum(1).max()
        else:
            trunc_ta_length = content.size(1)
        local_duration_pred = local_duration_pred[:, :trunc_ta_length]
        time_aligned_content = content[:, :trunc_ta_length]
        ta_content_mask = content_mask[:, :trunc_ta_length]
        local_duration_loss = loss_with_mask(
            (local_duration_target - local_duration_pred)**2,
            ta_content_mask,
            reduce=False
        )
        local_duration_loss *= is_time_aligned
        if is_time_aligned.sum().item() == 0:
            local_duration_loss *= 0.0
            local_duration_loss = local_duration_loss.mean()
        else:
            local_duration_loss = local_duration_loss.sum(
            ) / is_time_aligned.sum()
        global_duration_loss = F.mse_loss(
            global_duration_target, global_duration_pred
        )

        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0
                length_aligned_content[mask_indices] = 0

        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent
        )

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        if is_time_aligned.sum() == 0 and \
            duration.size(1) < content_mask.size(1):
            duration = F.pad(
                duration, (0, content_mask.size(1) - duration.size(1))
            )
        n_latents = torch.round(duration * self.autoencoder.latent_token_rate)
        helper_latent_mask = create_mask_from_length(n_latents.sum(1)).to(
            content_mask.device
        )
        attn_mask = ta_content_mask.unsqueeze(
            -1
        ) * helper_latent_mask.unsqueeze(1)
        align_path = create_alignment_path(n_latents, attn_mask)
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2).to(content.dtype), time_aligned_content
        )

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        # TODO compatility for 2D spectrogram VAE
        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        time_aligned_content = trim_or_pad_length(
            time_aligned_content, latent_length, 1
        )
        length_aligned_content = trim_or_pad_length(
            length_aligned_content, latent_length, 1
        )
        # time_aligned_content: from monotonic aligned input, without frame expansion (phoneme)
        # length_aligned_content: from aligned input (f0/energy)
        time_aligned_content = time_aligned_content + length_aligned_content
        time_aligned_content[~is_time_aligned] = self.dummy_ta_embed.to(
            time_aligned_content.dtype
        )
        context = content
        context[is_time_aligned] = self.dummy_nta_embed.to(context.dtype)
        # only use the first dummy non time aligned embedding
        context_mask[is_time_aligned, 1:] = False
        # truncate dummy non time aligned context
        if is_time_aligned.sum().item() < batch_size:
            trunc_nta_length = content_mask[~is_time_aligned].sum(1).max()
        else:
            trunc_nta_length = content.size(1)
        context = context[:, :trunc_nta_length]
        context_mask = context_mask[:, :trunc_nta_length]
        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            x_mask=latent_mask,
            timesteps=timesteps,
            context=context,
            context_mask=context_mask,
            time_aligned_context=time_aligned_content,
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = F.mse_loss(pred, target, reduction="none")
        diff_loss = loss_with_mask(diff_loss, latent_mask)
        return {
            "diff_loss": diff_loss,
            "local_duration_loss": local_duration_loss,
            "global_duration_loss": global_duration_loss,
        }

    def inference(
        self,
        content: list[Any],
        task: list[str],
        is_time_aligned: Sequence[bool],
        instruction: torch.Tensor,
        instruction_lengths: Sequence[int],
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        use_gt_duration: bool = False,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0

        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        length_aligned_content = content_output["length_aligned_content"]
        content, content_mask = content_output["content"], content_output[
            "content_mask"]
        context_mask = content_mask.detach()
        instruction_mask = create_mask_from_length(instruction_lengths)

        content, content_mask, global_duration_pred, local_duration_pred = \
            self.content_adapter(content, content_mask, instruction, instruction_mask)

        batch_size = content.size(0)

        # truncate dummy time aligned duration prediction
        is_time_aligned = torch.as_tensor(is_time_aligned)
        if is_time_aligned.sum() > 0:
            trunc_ta_length = content_mask[is_time_aligned].sum(1).max()
        else:
            trunc_ta_length = content.size(1)

        # prepare local duration
        local_duration_pred = torch.exp(local_duration_pred) * content_mask
        local_duration_pred = torch.ceil(
            local_duration_pred
        ) - self.duration_offset
        local_duration_pred = torch.round(local_duration_pred * self.frame_resolution * \
            self.autoencoder.latent_token_rate)
        local_duration_pred = local_duration_pred[:, :trunc_ta_length]
        # use ground truth duration
        if use_gt_duration and "duration" in kwargs:
            local_duration_pred = torch.round(
                torch.as_tensor(kwargs["duration"]) *
                self.autoencoder.latent_token_rate
            ).to(device)

        # prepare global duration
        global_duration = local_duration_pred.sum(1)
        global_duration_pred = torch.exp(
            global_duration_pred
        ) - self.duration_offset
        global_duration_pred *= self.autoencoder.latent_token_rate
        global_duration_pred = torch.round(global_duration_pred)
        global_duration[~is_time_aligned] = global_duration_pred[
            ~is_time_aligned]

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        time_aligned_content = content[:, :trunc_ta_length]
        ta_content_mask = content_mask[:, :trunc_ta_length]
        latent_mask = create_mask_from_length(global_duration).to(
            content_mask.device
        )
        attn_mask = ta_content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        align_path = create_alignment_path(local_duration_pred, attn_mask)
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2).to(content.dtype), time_aligned_content
        )  # (B, T, L) x (B, L, E) -> (B, T, E)
        time_aligned_content[~is_time_aligned] = self.dummy_ta_embed.to(
            time_aligned_content.dtype
        )

        length_aligned_content = trim_or_pad_length(
            length_aligned_content, time_aligned_content.size(1), 1
        )
        time_aligned_content = time_aligned_content + length_aligned_content

        # --------------------------------------------------------------------
        # prepare unconditional input
        # --------------------------------------------------------------------
        context = content
        context[is_time_aligned] = self.dummy_nta_embed.to(context.dtype)
        context_mask = content_mask
        context_mask[is_time_aligned, 1:] = False
        if is_time_aligned.sum().item() < batch_size:
            trunc_nta_length = content_mask[~is_time_aligned].sum(1).max()
        else:
            trunc_nta_length = content.size(1)
        context = context[:, :trunc_nta_length]
        context_mask = context_mask[:, :trunc_nta_length]

        if classifier_free_guidance:
            uncond_time_aligned_content = torch.zeros_like(
                time_aligned_content
            )
            uncond_context = torch.zeros_like(context)
            uncond_context_mask = context_mask.detach().clone()
            time_aligned_content = torch.cat(
                [uncond_time_aligned_content, time_aligned_content]
            )
            context = torch.cat([uncond_context, context])
            context_mask = torch.cat([uncond_context_mask, context_mask])
            latent_mask = torch.cat(
                [latent_mask, latent_mask.detach().clone()]
            )

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        latent_shape = tuple(
            int(global_duration.max().item()) if dim is None else dim
            for dim in self.autoencoder.latent_shape
        )
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=content.dtype
        )

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, num_steps = self.retrieve_timesteps(
            num_steps, device, timesteps=None, sigmas=sigmas
        )
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, timestep in enumerate(timesteps):
            if classifier_free_guidance:
                latent_input = torch.cat([latent, latent])
            else:
                latent_input = latent
            t = torch.tensor([timestep], device=device)
            t = t.expand(batch_size, )

            pred: torch.Tensor = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=t,
                context=context,
                context_mask=context_mask,
                time_aligned_context=time_aligned_content,
            )

            if classifier_free_guidance:
                pred_uncond, pred_cond = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            latent = self.infer_noise_scheduler.step(
                pred, timestep, latent
            ).prev_sample

            progress_bar.update(1)

        progress_bar.close()

        waveform = self.autoencoder.decode(latent)
        return waveform
