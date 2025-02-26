from typing import Sequence
import random
from typing import Any

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers.schedulers as noise_schedulers
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor

from models.autoencoder.autoencoder_base import AutoEncoderBase
from models.content_encoder.content_encoder import ContentEncoder
from models.common import LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase
from utils.torch_utilities import (
    create_alignment_path, create_mask_from_length, loss_with_mask,
    trim_or_pad_length
)


class DiffusionMixin:
    def __init__(
        self,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = True
    ) -> None:
        self.noise_scheduler_name = noise_scheduler_name
        self.snr_gamma = snr_gamma
        self.classifier_free_guidance = classifier_free_guidance
        self.noise_scheduler = noise_schedulers.DDPMScheduler.from_pretrained(
            self.noise_scheduler_name, subfolder="scheduler"
        )

    def compute_snr(self, timesteps) -> torch.Tensor:
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod)**0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device
                                                    )[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[...,
                                                                          None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma)**2
        return snr

    def get_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        training: bool = True
    ) -> torch.Tensor:
        if training:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size, ),
                device=device
            )
        else:
            # validation on half of the total timesteps
            timesteps = (self.noise_scheduler.config.num_train_timesteps //
                         2) * torch.ones((batch_size, ),
                                         dtype=torch.int64,
                                         device=device)

        timesteps = timesteps.long()
        return timesteps

    def get_target(
        self, latent: torch.Tensor, noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the target for loss depending on the prediction type
        """
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                latent, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )
        return target

    def loss_with_snr(
        self, pred: torch.Tensor, target: torch.Tensor,
        timesteps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.snr_gamma is None:
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss_with_mask(loss, mask)
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)],
                            dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss_with_mask(loss, mask, reduce=False) * mse_loss_weights
            loss = loss.mean()
        return loss


class AudioDiffusion(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    DiffusionMixin
):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = True,
    ):
        nn.Module.__init__(self)
        DiffusionMixin.__init__(
            self, noise_scheduler_name, snr_gamma, classifier_free_guidance
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
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

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
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        target = self.get_target(latent, noise, timesteps)

        pred = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )

        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        loss = self.loss_with_snr(pred, target, timesteps, latent_mask)

        return loss

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        scheduler: SchedulerMixin,
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
                content, task, device=device
            )
            content = content.repeat_interleave(num_samples_per_content, 0)
            content_mask = content_mask.repeat_interleave(
                num_samples_per_content, 0
            )

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        latent = self.prepare_latent(
            batch_size, scheduler, latent_shape, content.dtype, device
        )

        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, timestep in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            latent_input = torch.cat([latent, latent]
                                    ) if classifier_free_guidance else latent
            latent_input = scheduler.scale_model_input(latent_input, timestep)

            noise_pred = self.backbone(
                x=latent_input,
                timesteps=timestep,
                context=content,
                context_mask=content_mask
            )

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_content-noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i+1) % scheduler.order == 0):
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)

        return waveform

    def prepare_latent(
        self, batch_size: int, scheduler: SchedulerMixin,
        latent_shape: Sequence[int], dtype: torch.dtype, device: str
    ):
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=dtype
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latent = latent * scheduler.init_noise_sigma
        return latent

    def encode_content_classifier_free(
        self,
        content: list[Any],
        task: list[str],
        num_samples_per_content: int = 1
    ):
        device = self.dummy_param.device
        content, content_mask = self.content_encoder.encode_content(
            content, task, device
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


class VariableLengthAudioDiffusion(AudioDiffusion):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: nn.Module,
        backbone: nn.Module,
        # latent_dim: int,
        frame_resolution:
        float,  # frame resolution in second for duration prediction
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = True,
    ):
        super().__init__(
            autoencoder, content_encoder, backbone, noise_scheduler_name,
            snr_gamma, classifier_free_guidance
        )
        self.content_adapter = content_adapter
        self.frame_resolution = frame_resolution
        # self.backbone_in_proj = nn.Linear(
        #     latent_dim + self.content_adapter.d_out, latent_dim
        # )
        self.dummy_nta_embed = nn.Parameter(torch.zeros(content_adapter.d_out))

    def forward(
        self, content, condition, duration, duration_lengths, task, waveform,
        waveform_lengths, **kwargs
    ):
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

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
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)

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
        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        if time_aligned_content.size(1) > latent_length:
            time_aligned_content = time_aligned_content[:, :latent_length]
        elif time_aligned_content.size(1) < latent_length:
            pad_size = latent_length - time_aligned_content.size(1)
            padding = (0, 0, 0,
                       pad_size) + (0, 0) * (time_aligned_content.ndim - 2)
            time_aligned_content = F.pad(
                time_aligned_content, padding, mode="constant", value=0
            )

        # latent_input = torch.cat(
        #     [
        #         noisy_latent.transpose(1, self.autoencoder.time_dim),
        #         time_aligned_content
        #     ],
        #     dim=2,
        # )
        # latent_input = self.backbone_in_proj(latent_input)
        # latent_input = latent_input.transpose(1, self.autoencoder.time_dim)

        context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                       (..., )].expand_as(content)
        context_mask = torch.ones(batch_size, context.size(1)).to(device)
        pred: torch.Tensor = self.backbone(
            # x=latent_input,
            x=noisy_latent,
            timesteps=timesteps,
            time_aligned_context=time_aligned_content,
            context=context,
            x_mask=latent_mask,
            context_mask=context_mask
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = self.loss_with_snr(pred, target, timesteps, latent_mask)
        return {
            "diff_loss": diff_loss,
            "local_duration_loss": local_duration_loss
        }

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        scheduler: SchedulerMixin,
        num_steps: int = 20,
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

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps
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
        # scale the initial noise by the standard deviation required by the scheduler
        latent = latent * scheduler.init_noise_sigma

        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        # --------------------------------------------------------------------
        # iteratively denoising
        # --------------------------------------------------------------------
        for i, timestep in enumerate(timesteps):
            # latent_input = torch.cat(
            #     [
            #         latent.transpose(1, self.autoencoder.time_dim),
            #         time_aligned_content
            #     ],
            #     dim=2,
            # )
            # latent_input = self.backbone_in_proj(latent_input)
            # latent_input = latent_input.transpose(1, self.autoencoder.time_dim)

            latent_input = scheduler.scale_model_input(latent, timestep)

            context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                           (..., )].expand_as(content)
            context_mask = torch.ones(batch_size, context.size(1)).to(device)

            noise_pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=timestep,
                time_aligned_context=time_aligned_content,
                context=context,
                context_mask=context_mask
            )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i+1) % scheduler.order == 0):
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)
        return waveform


class SameLengthAudioDiffusion(VariableLengthAudioDiffusion):
    def forward(
        self, content, condition, duration, duration_lengths, task, waveform,
        waveform_lengths, **kwargs
    ):
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        content_input = content
        # content: (B, L, E)
        content, content_mask = self.content_encoder.encode_content(
            content_input, task, device=device
        )
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,
        ) = self.content_adapter(content, content_mask)

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
        # prepare latent and diffusion-related noise
        # --------------------------------------------------------------------
        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < 0.1
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        batch_size = latent.shape[0]
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)

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
        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        time_aligned_content1 = trim_or_pad_length(
            time_aligned_content1, latent_length, 1
        )
        time_aligned_content2 = trim_or_pad_length(
            time_aligned_content2, latent_length, 1
        )
        # time_aligned_content1: from unaligned input (phoneme)
        # time_aligned_content2: from aligned input (f0/energy)
        time_aligned_content = time_aligned_content1 + time_aligned_content2

        context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                       (..., )].expand_as(content)
        context_mask = torch.ones(batch_size, context.size(1)).to(device)
        pred: torch.Tensor = self.backbone(
            # x=latent_input,
            x=noisy_latent,
            timesteps=timesteps,
            time_aligned_context=time_aligned_content,
            context=context,
            x_mask=latent_mask,
            context_mask=context_mask
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = self.loss_with_snr(pred, target, timesteps, latent_mask)
        return {
            "diff_loss": diff_loss,
            "local_duration_loss": local_duration_loss
        }

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        scheduler: SchedulerMixin,
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        use_gt_duration: bool = False,
        **kwargs
    ):
        device = self.dummy_param.device

        content_input = content
        content, content_mask = self.content_encoder.encode_content(
            content_input, task, device=device
        )
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,  # log(n_frames)
        ) = self.content_adapter(content, content_mask)

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        local_duration = torch.round(
            torch.as_tensor(kwargs["duration"]) *
            self.autoencoder.latent_token_rate
        ).to(device)

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        # content_mask: [B, L], latent_mask: [B, T]
        global_duration = local_duration.sum(1)
        latent_mask = create_mask_from_length(global_duration).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        align_path = create_alignment_path(local_duration_pred, attn_mask)
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
        # scale the initial noise by the standard deviation required by the scheduler
        latent = latent * scheduler.init_noise_sigma

        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        # --------------------------------------------------------------------
        # iteratively denoising
        # --------------------------------------------------------------------
        for i, timestep in enumerate(timesteps):

            latent_input = scheduler.scale_model_input(latent, timestep)

            context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
                                           (..., )].expand_as(content)
            context_mask = torch.ones(batch_size, context.size(1)).to(device)

            noise_pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=timestep,
                time_aligned_context=time_aligned_content,
                context=context,
                context_mask=context_mask
            )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i+1) % scheduler.order == 0):
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)
        return waveform


class DiffSingerDiffusion(VariableLengthAudioDiffusion):
    def forward(
        self, content, condition, duration, duration_lengths, task, waveform,
        waveform_lengths, **kwargs
    ):
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

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

        duration = torch.round(duration * self.autoencoder.latent_token_rate)
        local_duration_target = torch.log(duration + 1e-8)
        global_duration_target = torch.log(
            latent_mask.sum(1) // self.autoencoder.latent_token_rate + 1e-8
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
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        # content_mask: [B, L], helper_latent_mask: [B, T]
        helper_latent_mask = create_mask_from_length(duration.sum(1)).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1
                                          ) * helper_latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        align_path = create_alignment_path(duration, attn_mask)
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2), content
        )  # (B, T, L) x (B, L, E) -> (B, T, E)

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        # TODO compatility for 2D spectrogram VAE
        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        if time_aligned_content.size(1) > latent_length:
            time_aligned_content = time_aligned_content[:, :latent_length]
        elif time_aligned_content.size(1) < latent_length:
            pad_size = latent_length - time_aligned_content.size(1)
            padding = (0, 0, 0,
                       pad_size) + (0, 0) * (time_aligned_content.ndim - 2)
            time_aligned_content = F.pad(
                time_aligned_content, padding, mode="constant", value=0
            )

        pred = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            context=time_aligned_content.transpose(1, 2),
            x_mask=latent_mask,
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = self.loss_with_snr(pred, target, timesteps, latent_mask)
        loss = diff_loss + local_duration_loss
        return loss

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        scheduler: SchedulerMixin,
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
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
            local_duration_pred,
        ) = self.content_adapter(content, content_mask)

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps
        local_duration_pred = torch.exp(local_duration_pred) * content_mask
        local_duration_pred = torch.ceil(local_duration_pred)

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
        # scale the initial noise by the standard deviation required by the scheduler
        latent = latent * scheduler.init_noise_sigma

        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        # --------------------------------------------------------------------
        # iteratively denoising
        # --------------------------------------------------------------------
        for i, timestep in enumerate(timesteps):

            latent = scheduler.scale_model_input(latent, timestep)
            noise_pred = self.backbone(
                x=latent,
                timesteps=timestep,
                context=time_aligned_content.transpose(1, 2),
                x_mask=latent_mask,
            )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i+1) % scheduler.order == 0):
                progress_bar.update(1)

        waveform = self.autoencoder.decode(latent)
        return waveform
