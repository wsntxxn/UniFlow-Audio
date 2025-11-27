from typing import Any, Optional, Union, List, Sequence

import inspect
import random

from tqdm import tqdm
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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
from constants import SAME_LENGTH_TASKS


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
        training: bool,
    ):
        batch_size = latent.shape[0]
        noise = torch.randn_like(latent)

        if training:
            if self.sample_strategy == 'normal':
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=batch_size,
                    logit_mean=0,
                    logit_std=1,
                    mode_scale=None,
                )
            elif self.sample_strategy == 'uniform':
                u = torch.rand(batch_size, )
            else:
                raise NotImplementedError(
                    f"{self.sample_strategy} samlping for timesteps is not supported now"
                )

            indices = (
                u * self.train_noise_scheduler.config.num_train_timesteps
            ).long()
        else:
            indices = (
                self.train_noise_scheduler.config.num_train_timesteps // 2
            ) * torch.ones((batch_size, )).long()

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
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

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


class ContentEncoderAdapterMixin:
    def __init__(
        self,
        content_encoder: ContentEncoder,
        content_adapter: ContentAdapterBase | None = None
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
        content, content_mask = content_output["content"], content_output[
            "content_mask"]

        if instruction is not None:
            instruction_mask = create_mask_from_length(instruction_lengths)
            (
                content,
                content_mask,
                global_duration_pred,
                local_duration_pred,
            ) = self.content_adapter(
                content, content_mask, instruction, instruction_mask
            )

        return_dict = {
            "content": content,
            "content_mask": content_mask,
            "length_aligned_content": content_output["length_aligned_content"],
        }
        if instruction is not None:
            return_dict["global_duration_pred"] = global_duration_pred
            return_dict["local_duration_pred"] = local_duration_pred

        return return_dict


class SingleTaskCrossAttentionAudioFlowMatching(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    FlowMatchingMixin, ContentEncoderAdapterMixin
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
        ContentEncoderAdapterMixin.__init__(
            self, content_encoder=content_encoder
        )

        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        if hasattr(
            self.content_encoder, "audio_encoder"
        ) and self.content_encoder.audio_encoder is not None:
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

        content_dict = self.encode_content(content, task, device)
        content, content_mask = content_dict["content"], content_dict[
            "content_mask"]

        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent, training=self.training
        )

        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )

        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        loss = loss_with_mask(loss, latent_mask)

        return loss

    def iterative_denoise(
        self, latent: torch.Tensor, timesteps: list[int], num_steps: int,
        verbose: bool, cfg: bool, cfg_scale: float, backbone_input: dict
    ):
        progress_bar = tqdm(range(num_steps), disable=not verbose)

        for i, timestep in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            if cfg:
                latent_input = torch.cat([latent, latent])
            else:
                latent_input = latent

            noise_pred: torch.Tensor = self.backbone(
                x=latent_input, timesteps=timestep, **backbone_input
            )

            # perform guidance
            if cfg:
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_content - noise_pred_uncond
                )

            latent = self.infer_noise_scheduler.step(
                noise_pred, timestep, latent
            ).prev_sample

            progress_bar.update(1)

        progress_bar.close()

        return latent

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        latent_shape: Sequence[int],
        num_steps: int = 50,
        sway_sampling_coef: float | None = -1.0,
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

        if not sway_sampling_coef:
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        else:
            t = torch.linspace(0, 1, num_steps + 1)
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            sigmas = 1 - t
        timesteps, num_steps = self.retrieve_timesteps(
            num_steps, device, timesteps=None, sigmas=sigmas
        )

        latent = self.iterative_denoise(
            latent=latent,
            timesteps=timesteps,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "context": content,
                "context_mask": content_mask,
            },
        )

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
        content_dict = self.content_encoder.encode_content(
            content, task, device=device
        )
        content, content_mask = content_dict["content"], content_dict[
            "content_mask"]

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


class DurationAdapterMixin:
    def __init__(
        self,
        latent_token_rate: int,
        offset: float = 1.0,
        frame_resolution: float | None = None
    ):
        self.latent_token_rate = latent_token_rate
        self.offset = offset
        self.frame_resolution = frame_resolution

    def get_global_duration_loss(
        self,
        pred: torch.Tensor,
        latent_mask: torch.Tensor,
        reduce: bool = True,
    ):
        target = torch.log(
            latent_mask.sum(1) / self.latent_token_rate + self.offset
        )
        loss = F.mse_loss(target, pred, reduction="mean" if reduce else "none")
        return loss

    def get_local_duration_loss(
        self, ground_truth: torch.Tensor, pred: torch.Tensor,
        mask: torch.Tensor, is_time_aligned: Sequence[bool], reduce: bool
    ):
        n_frames = torch.round(ground_truth / self.frame_resolution)
        target = torch.log(n_frames + self.offset)
        loss = loss_with_mask(
            (target - pred)**2,
            mask,
            reduce=False,
        )
        loss *= is_time_aligned
        if reduce:
            if is_time_aligned.sum().item() == 0:
                loss *= 0.0
                loss = loss.mean()
            else:
                loss = loss.sum() / is_time_aligned.sum()

        return loss

    def prepare_local_duration(self, pred: torch.Tensor, mask: torch.Tensor):
        pred = torch.exp(pred) * mask
        pred = torch.ceil(pred) - self.offset
        pred *= self.frame_resolution
        return pred

    def prepare_global_duration(
        self,
        global_pred: torch.Tensor,
        local_pred: torch.Tensor,
        is_time_aligned: Sequence[bool],
        use_local: bool = True,
    ):
        """
        global_pred: predicted duration value, processed by logarithmic and offset
        local_pred: predicted latent length 
        """
        global_pred = torch.exp(global_pred) - self.offset
        result = global_pred
        # avoid error accumulation for each frame
        if use_local:
            pred_from_local = torch.round(local_pred * self.latent_token_rate)
            pred_from_local = pred_from_local.sum(1) / self.latent_token_rate
            result[is_time_aligned] = pred_from_local[is_time_aligned]

        return result

    def expand_by_duration(
        self,
        x: torch.Tensor,
        content_mask: torch.Tensor,
        local_duration: torch.Tensor,
        global_duration: torch.Tensor,
    ):
        training = getattr(self, 'training', False)
        n_latents = torch.round(local_duration * self.latent_token_rate)
        # TODO unify duration input scale for training and inference: num_frames
        if not training:  # inference mode
            latent_length = torch.round(
                global_duration * self.latent_token_rate
            )
        else:  # training mode
            latent_length = n_latents.sum(1)
        latent_mask = create_mask_from_length(latent_length).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)
        align_path = create_alignment_path(n_latents, attn_mask)
        expanded_x = torch.matmul(align_path.transpose(1, 2).to(x.dtype), x)
        return expanded_x, latent_mask


class CrossAttentionAudioFlowMatching(
    SingleTaskCrossAttentionAudioFlowMatching, DurationAdapterMixin
):
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
            backbone=backbone,
            cfg_drop_ratio=cfg_drop_ratio,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps,
        )
        ContentEncoderAdapterMixin.__init__(
            self,
            content_encoder=content_encoder,
            content_adapter=content_adapter
        )
        DurationAdapterMixin.__init__(
            self,
            latent_token_rate=autoencoder.latent_token_rate,
            offset=duration_offset
        )

    def encode_content_with_instruction(
        self, content: list[Any], task: list[str], device,
        instruction: torch.Tensor, instruction_lengths: torch.Tensor
    ):
        content_dict = self.encode_content(
            content, task, device, instruction, instruction_lengths
        )
        return (
            content_dict["content"],
            content_dict["content_mask"],
            content_dict["global_duration_pred"],
            content_dict["local_duration_pred"],
            content_dict["length_aligned_content"],
        )

    def forward(
        self,
        content: list[Any],
        task: list[str],
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        instruction: torch.Tensor,
        instruction_lengths: torch.Tensor,
        loss_reduce: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        loss_reduce = self.training or (loss_reduce and not self.training)

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        content, content_mask, global_duration_pred, _, _ = \
            self.encode_content_with_instruction(
                content, task, device, instruction, instruction_lengths
            )

        global_duration_loss = self.get_global_duration_loss(
            global_duration_pred, latent_mask, reduce=loss_reduce
        )

        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent, training=self.training
        )

        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask,
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        diff_loss = loss_with_mask(diff_loss, latent_mask, reduce=loss_reduce)

        return {
            "diff_loss": diff_loss,
            "global_duration_loss": global_duration_loss,
        }

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        is_time_aligned: Sequence[bool],
        instruction: torch.Tensor,
        instruction_lengths: torch.Tensor,
        num_steps: int = 20,
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        disable_progress=True,
        use_gt_duration: bool = False,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0

        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,
            _,
        ) = self.encode_content_with_instruction(
            content, task, device, instruction, instruction_lengths
        )
        batch_size = content.size(0)

        if use_gt_duration:
            raise NotImplementedError(
                "Using ground truth global duration only is not implemented yet"
            )

        # prepare global duration
        global_duration = self.prepare_global_duration(
            global_duration_pred,
            local_duration_pred,
            is_time_aligned,
            use_local=False
        )
        # TODO: manually set duration for SE and AudioSR
        latent_length = torch.round(global_duration * self.latent_token_rate)
        task_mask = torch.as_tensor([t in SAME_LENGTH_TASKS for t in task])
        latent_length[task_mask] = content[task_mask].size(1)
        latent_mask = create_mask_from_length(latent_length).to(device)
        max_latent_length = latent_mask.sum(1).max().item()

        # prepare latent and noise
        if classifier_free_guidance:
            uncond_context = torch.zeros_like(content)
            uncond_content_mask = content_mask.detach().clone()
            context = torch.cat([uncond_context, content])
            context_mask = torch.cat([uncond_content_mask, content_mask])
        else:
            context = content
            context_mask = content_mask

        latent_shape = tuple(
            max_latent_length if dim is None else dim
            for dim in self.autoencoder.latent_shape
        )
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=content.dtype
        )
        if not sway_sampling_coef:
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        else:
            t = torch.linspace(0, 1, num_steps + 1)
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            sigmas = 1 - t
        timesteps, num_steps = self.retrieve_timesteps(
            num_steps, device, timesteps=None, sigmas=sigmas
        )
        latent = self.iterative_denoise(
            latent=latent,
            timesteps=timesteps,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "x_mask": latent_mask,
                "context": context,
                "context_mask": context_mask,
            }
        )

        waveform = self.autoencoder.decode(latent)
        return waveform


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
        num_train_steps: int = 1000,
        task_weights: dict | None = None,
    ):

        super().__init__(
            autoencoder=autoencoder,
            content_encoder=content_encoder,
            content_adapter=content_adapter,
            backbone=backbone,
            content_dim=content_dim,
            frame_resolution=frame_resolution,
            duration_offset=duration_offset,
            cfg_drop_ratio=cfg_drop_ratio,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps
        )
        DurationAdapterMixin.__init__(
            self,
            latent_token_rate=autoencoder.latent_token_rate,
            offset=duration_offset,
            frame_resolution=frame_resolution
        )
        self.dummy_nta_embed = nn.Parameter(torch.zeros(content_dim))
        self.dummy_ta_embed = nn.Parameter(torch.zeros(content_dim))
        self.task_weights = task_weights

    def get_backbone_input(
        self, target_length: int, content: torch.Tensor,
        content_mask: torch.Tensor, time_aligned_content: torch.Tensor,
        length_aligned_content: torch.Tensor, is_time_aligned: torch.Tensor
    ):
        # TODO compatility for 2D spectrogram VAE
        time_aligned_content = trim_or_pad_length(
            time_aligned_content, target_length, 1
        )
        length_aligned_content = trim_or_pad_length(
            length_aligned_content, target_length, 1
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
        context_mask = content_mask.detach().clone()
        context_mask[is_time_aligned, 1:] = False

        # truncate dummy non time aligned context
        if is_time_aligned.sum().item() < content.size(0):
            trunc_nta_length = content_mask[~is_time_aligned].sum(1).max()
        else:
            trunc_nta_length = content.size(1)
        context = context[:, :trunc_nta_length]
        context_mask = context_mask[:, :trunc_nta_length]

        return context, context_mask, time_aligned_content

    def forward(
        self,
        content: list[Any],
        duration: Sequence[float],
        task: list[str],
        is_time_aligned: Sequence[bool],
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        instruction: torch.Tensor,
        instruction_lengths: torch.Tensor,
        loss_reduce: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        if self.training:
            if self.task_weights:
                loss_reduce = False
            else:
                loss_reduce = True
        # loss_reduce = self.training or (loss_reduce and not self.training)

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        (
            content, content_mask, global_duration_pred, local_duration_pred,
            length_aligned_content
        ) = self.encode_content_with_instruction(
            content, task, device, instruction, instruction_lengths
        )

        # truncate unused non time aligned duration prediction
        if is_time_aligned.sum() > 0:
            trunc_ta_length = content_mask[is_time_aligned].sum(1).max()
        else:
            trunc_ta_length = content.size(1)

        # duration loss
        local_duration_pred = local_duration_pred[:, :trunc_ta_length]
        ta_content_mask = content_mask[:, :trunc_ta_length]
        local_duration_loss = self.get_local_duration_loss(
            duration,
            local_duration_pred,
            ta_content_mask,
            is_time_aligned,
            reduce=loss_reduce
        )

        global_duration_loss = self.get_global_duration_loss(
            global_duration_pred, latent_mask, reduce=loss_reduce
        )

        # --------------------------------------------------------------------
        # prepare latent and noise
        # --------------------------------------------------------------------
        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent, training=self.training
        )

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        if is_time_aligned.sum() == 0 and \
            duration.size(1) < content_mask.size(1):
            duration = F.pad(
                duration, (0, content_mask.size(1) - duration.size(1))
            )
        time_aligned_content, _ = self.expand_by_duration(
            x=content[:, :trunc_ta_length],
            content_mask=ta_content_mask,
            local_duration=duration,
            global_duration=latent_mask.sum(1),
        )

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        # TODO compatility for 2D spectrogram VAE
        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        context, context_mask, time_aligned_content = self.get_backbone_input(
            latent_length, content, content_mask, time_aligned_content,
            length_aligned_content, is_time_aligned
        )

        # --------------------------------------------------------------------
        # classifier free guidance
        # --------------------------------------------------------------------
        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                context[mask_indices] = 0
                time_aligned_content[mask_indices] = 0

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
        diff_loss = loss_with_mask(diff_loss, latent_mask, reduce=loss_reduce)

        if self.training and self.task_weights:
            loss_weights = torch.tensor([self.task_weights[t] for t in task],
                                        device=device)
            diff_loss = (diff_loss * loss_weights).sum() / loss_weights.sum()
            local_duration_loss = (local_duration_loss *
                                   loss_weights).sum() / loss_weights.sum()
            global_duration_loss = (global_duration_loss *
                                    loss_weights).sum() / loss_weights.sum()

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
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        use_gt_duration: bool = False,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0

        (
            content, content_mask, global_duration_pred, local_duration_pred,
            length_aligned_content
        ) = self.encode_content_with_instruction(
            content, task, device, instruction, instruction_lengths
        )
        # print("content std: ", content.std())
        batch_size = content.size(0)

        # truncate dummy time aligned duration prediction
        is_time_aligned = torch.as_tensor(is_time_aligned)
        if is_time_aligned.sum() > 0:
            trunc_ta_length = content_mask[is_time_aligned].sum(1).max()
        else:
            trunc_ta_length = content.size(1)

        # prepare local duration
        local_duration = self.prepare_local_duration(
            local_duration_pred, content_mask
        )
        local_duration = local_duration[:, :trunc_ta_length]
        # use ground truth duration
        if use_gt_duration and "duration" in kwargs:
            local_duration = torch.as_tensor(kwargs["duration"]).to(device)

        # prepare global duration
        global_duration = self.prepare_global_duration(
            global_duration_pred, local_duration, is_time_aligned
        )

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        time_aligned_content, latent_mask = self.expand_by_duration(
            x=content[:, :trunc_ta_length],
            content_mask=content_mask[:, :trunc_ta_length],
            local_duration=local_duration,
            global_duration=global_duration,
        )

        context, context_mask, time_aligned_content = self.get_backbone_input(
            target_length=time_aligned_content.size(1),
            content=content,
            content_mask=content_mask,
            time_aligned_content=time_aligned_content,
            length_aligned_content=length_aligned_content,
            is_time_aligned=is_time_aligned
        )

        # --------------------------------------------------------------------
        # prepare unconditional input
        # --------------------------------------------------------------------
        if classifier_free_guidance:
            uncond_time_aligned_content = torch.zeros_like(
                time_aligned_content
            )
            uncond_context = torch.zeros_like(context)
            uncond_context_mask = context_mask.detach().clone()
            time_aligned_content = torch.cat([
                uncond_time_aligned_content, time_aligned_content
            ])
            context = torch.cat([uncond_context, context])
            context_mask = torch.cat([uncond_context_mask, context_mask])
            latent_mask = torch.cat([
                latent_mask, latent_mask.detach().clone()
            ])

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        latent_length = latent_mask.sum(1).max().item()
        latent_shape = tuple(
            latent_length if dim is None else dim
            for dim in self.autoencoder.latent_shape
        )
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=content.dtype
        )

        if not sway_sampling_coef:
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        else:
            t = torch.linspace(0, 1, num_steps + 1)
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            sigmas = 1 - t
        timesteps, num_steps = self.retrieve_timesteps(
            num_steps, device, timesteps=None, sigmas=sigmas
        )
        latent = self.iterative_denoise(
            latent=latent,
            timesteps=timesteps,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "x_mask": latent_mask,
                "context": context,
                "context_mask": context_mask,
                "time_aligned_context": time_aligned_content,
            }
        )

        waveform = self.autoencoder.decode(latent)
        return waveform


class DoubleContentAudioFlowMatching(DummyContentAudioFlowMatching):
    def get_backbone_input(
        self, target_length: int, content: torch.Tensor,
        content_mask: torch.Tensor, time_aligned_content: torch.Tensor,
        length_aligned_content: torch.Tensor, is_time_aligned: torch.Tensor
    ):
        # TODO compatility for 2D spectrogram VAE
        time_aligned_content = trim_or_pad_length(
            time_aligned_content, target_length, 1
        )
        context_length = min(content.size(1), time_aligned_content.size(1))
        time_aligned_content[~is_time_aligned, :context_length] = content[
            ~is_time_aligned, :context_length]
        length_aligned_content = trim_or_pad_length(
            length_aligned_content, target_length, 1
        )
        # time_aligned_content: from monotonic aligned input, without frame expansion (phoneme)
        # length_aligned_content: from aligned input (f0/energy)
        time_aligned_content = time_aligned_content + length_aligned_content

        context = content
        context_mask = content_mask.detach().clone()

        return context, context_mask, time_aligned_content


class HybridContentAudioFlowMatching(DummyContentAudioFlowMatching):
    def get_backbone_input(
        self, target_length: int, content: torch.Tensor,
        content_mask: torch.Tensor, time_aligned_content: torch.Tensor,
        length_aligned_content: torch.Tensor, is_time_aligned: torch.Tensor
    ):
        # TODO compatility for 2D spectrogram VAE
        time_aligned_content = trim_or_pad_length(
            time_aligned_content, target_length, 1
        )
        length_aligned_content = trim_or_pad_length(
            length_aligned_content, target_length, 1
        )
        # time_aligned_content: from monotonic aligned input, without frame expansion (phoneme)
        # length_aligned_content: from aligned input (f0/energy)
        time_aligned_content = time_aligned_content + length_aligned_content
        time_aligned_content[~is_time_aligned] = self.dummy_ta_embed.to(
            time_aligned_content.dtype
        )

        context = content
        context_mask = content_mask.detach().clone()

        return context, context_mask, time_aligned_content


class DummyContentFillerAudioFlowMatching(DummyContentAudioFlowMatching):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: ContentAdapterBase,
        backbone: nn.Module,
        content_dim: int,
        frame_resolution: float,
        duration_offset: float = 1,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000
    ):
        super().__init__(
            autoencoder, content_encoder, content_adapter, backbone,
            content_dim, frame_resolution, duration_offset, cfg_drop_ratio,
            sample_strategy, num_train_steps
        )
        self.filler_embedding = nn.Parameter(torch.empty(content_dim))
        init.normal_(self.filler_embedding)

    def expand_by_duration(
        self,
        x: torch.Tensor,
        content_mask: torch.Tensor,
        local_duration: torch.Tensor | None,
        global_duration: torch.Tensor,
    ):
        """
        x: content tensor, [batch_size, content_length, content_dim]
        content_mask: content mask, [batch_size, content_length]
        local_duration: content element length, not used in this model
        global_duration: clip length in latent tokens during training; clip duration in seconds during inference
        """
        if not self.training:
            global_duration = torch.round(
                global_duration * self.latent_token_rate
            ).long()
        if x.size(1) < global_duration.max():
            padding_length = global_duration.max() - x.size(1)
            padding = torch.zeros(x.size(0), padding_length,
                                  x.size(2)).to(x.device).to(x.dtype)
            x = torch.cat([x, padding], dim=1)

        filling_mask = torch.ones(
            x.size(0), global_duration.max(), dtype=bool
        ).to(x.device)
        filling_mask[:, :content_mask.size(1)][content_mask] = 0
        filling_mask = filling_mask.unsqueeze(-1)
        num_filling = filling_mask.sum()
        filler_embedding = self.filler_embedding.to(dtype=x.dtype)
        x = x.masked_scatter(
            filling_mask, filler_embedding.repeat(num_filling, 1)
        )

        if self.training:
            latent_mask = None
        else:
            latent_length = global_duration
            latent_mask = create_mask_from_length(latent_length).to(x.device)
        return x, latent_mask
