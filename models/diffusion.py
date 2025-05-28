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
from models.content_adapter import ContentAdapterBase
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
        cfg_drop_ratio: float = 0.2
    ) -> None:
        self.noise_scheduler_name = noise_scheduler_name
        self.snr_gamma = snr_gamma
        self.classifier_free_guidance = cfg_drop_ratio > 0.0
        self.cfg_drop_ratio = cfg_drop_ratio
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
            # Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py#L1006
            snr = self.compute_snr(timesteps)
            mse_loss_weights = torch.stack(
                [
                    snr,
                    self.snr_gamma * torch.ones_like(timesteps),
                ],
                dim=1,
            ).min(dim=1)[0]
            # division by (snr + 1) does not work well, not clear about the reason
            mse_loss_weights = mse_loss_weights / snr
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss_with_mask(loss, mask, reduce=False) * mse_loss_weights
            loss = loss.mean()
        return loss

    def rescale_cfg(
        self, pred_cond: torch.Tensor, pred_cfg: torch.Tensor,
        guidance_rescale: float
    ):
        """
        Rescale `pred_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_cond = pred_cond.std(
            dim=list(range(1, pred_cond.ndim)), keepdim=True
        )
        std_cfg = pred_cfg.std(dim=list(range(1, pred_cfg.ndim)), keepdim=True)

        pred_rescaled = pred_cfg * (std_cond / std_cfg)
        pred_cfg = guidance_rescale * pred_rescaled + (
            1 - guidance_rescale
        ) * pred_cfg
        return pred_cfg


class CrossAttentionAudioDiffusion(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    DiffusionMixin
):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: ContentAdapterBase,
        backbone: nn.Module,
        duration_offset: float = 1.0,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        cfg_drop_ratio: float = 0.2,
    ):
        nn.Module.__init__(self)
        DiffusionMixin.__init__(
            self, noise_scheduler_name, snr_gamma, cfg_drop_ratio
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
        instruction_lengths: Sequence[int], **kwargs
    ):
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        content, content_mask = content_output["content"], content_output[
            "content_mask"]
        instruction_mask = create_mask_from_length(instruction_lengths)
        content, content_mask, global_duration_pred, _ = \
            self.content_adapter(content, content_mask, instruction, instruction_mask)
        global_duration_target = torch.log(
            latent_mask.sum(1) / self.autoencoder.latent_token_rate +
            self.duration_offset
        )
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

        batch_size = latent.shape[0]
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)

        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )

        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = self.loss_with_snr(pred, target, timesteps, latent_mask)

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
        instruction: torch.Tensor,
        instruction_lengths: Sequence[int],
        scheduler: SchedulerMixin,
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        disable_progress: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0

        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        content, content_mask = content_output["content"], content_output[
            "content_mask"]

        instruction_mask = create_mask_from_length(instruction_lengths)
        content, content_mask, global_duration_pred, _ = \
            self.content_adapter(content, content_mask, instruction, instruction_mask)

        if classifier_free_guidance:
            uncond_content = torch.zeros_like(content)
            uncond_content_mask = content_mask.detach().clone()
            content = torch.cat([uncond_content, content])
            content_mask = torch.cat([uncond_content_mask, content_mask])

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        global_duration_pred = torch.exp(
            global_duration_pred
        ) - self.duration_offset
        global_duration_pred *= self.autoencoder.latent_token_rate
        global_duration_pred = torch.round(global_duration_pred)
        batch_size = content.size(0)
        latent_shape = tuple(
            int(global_duration_pred.max().item()) if dim is None else dim
            for dim in self.autoencoder.latent_shape
        )
        latent = self.prepare_latent(
            batch_size, scheduler, latent_shape, content.dtype, device
        )
        latent_mask = create_mask_from_length(global_duration_pred).to(
            content_mask.device
        )
        if classifier_free_guidance:
            latent_mask = torch.cat([latent_mask, latent_mask])

        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, timestep in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            latent_input = torch.cat([latent, latent]
                                    ) if classifier_free_guidance else latent
            latent_input = scheduler.scale_model_input(latent_input, timestep)

            noise_pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=timestep,
                context=content,
                context_mask=content_mask,
            )

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_content - noise_pred_uncond
                )
                if guidance_rescale != 0.0:
                    noise_pred = self.rescale_cfg(
                        noise_pred_content, noise_pred, guidance_rescale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i + 1) % scheduler.order == 0):
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


class DummyContentAudioDiffusion(CrossAttentionAudioDiffusion):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: ContentAdapterBase,
        backbone: nn.Module,
        content_dim: int,
        frame_resolution: float,
        duration_offset: float = 1.0,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        cfg_drop_ratio: float = 0.2,
    ):
        """
        Args:
            autoencoder:
                Pretrained audio autoencoder that encodes raw waveforms into latent
                space and decodes latents back to waveforms.
            content_encoder:
                Module that produces content embeddings (e.g., from text, MIDI, or
                other modalities) used to guide the diffusion.
            content_adapter (ContentAdapterBase):
                Adapter module that fuses task instruction embeddings and content embeddings,
                and performs duration prediction for time-aligned tasks.
            backbone:
                U‑Net or Transformer backbone that performs the core denoising
                operations in latent space.
            content_dim:
                Dimension of the content embeddings produced by the `content_encoder` 
                and `content_adapter`.
            frame_resolution:
                Time resolution, in seconds, of each content frame when predicting
                duration alignment. Used when calculating duration loss.
            duration_offset:
                A small positive offset (frame number) added to predicted durations
                to ensure numerical stability of log-scaled duration prediction. 
            noise_scheduler_name:
                Identifier of the pretrained noise scheduler to use. 
            snr_gamma:
                Clipping value in min-SNR diffusion loss weighting strategy.
            cfg_drop_ratio:
                Probability of dropping the content conditioning during training
                to support CFG.
        """
        super().__init__(
            autoencoder=autoencoder,
            content_encoder=content_encoder,
            content_adapter=content_adapter,
            backbone=backbone,
            duration_offset=duration_offset,
            noise_scheduler_name=noise_scheduler_name,
            snr_gamma=snr_gamma,
            cfg_drop_ratio=cfg_drop_ratio,
        )
        self.frame_resolution = frame_resolution
        self.dummy_nta_embed = nn.Parameter(torch.zeros(content_dim))
        self.dummy_ta_embed = nn.Parameter(torch.zeros(content_dim))

    def forward(
        self, content, duration, task, is_time_aligned, waveform,
        waveform_lengths, instruction, instruction_lengths, **kwargs
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
        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        length_aligned_content = content_output["length_aligned_content"]
        content, content_mask = content_output["content"], content_output[
            "content_mask"]
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

        # local duration loss
        local_duration_pred = local_duration_pred[:, :trunc_ta_length]
        ta_content_mask = content_mask[:, :trunc_ta_length]
        local_duration_target = local_duration_target.to(
            dtype=local_duration_pred.dtype
        )
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

        # global duration loss
        global_duration_loss = F.mse_loss(
            global_duration_target, global_duration_pred
        )

        # --------------------------------------------------------------------
        # prepare latent and diffusion-related noise
        # --------------------------------------------------------------------

        batch_size = latent.shape[0]
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        if is_time_aligned.sum() == 0 and \
            duration.size(1) < content_mask.size(1):
            # for non time-aligned tasks like TTA, `duration` is dummy one
            duration = F.pad(
                duration, (0, content_mask.size(1) - duration.size(1))
            )
        n_latents = torch.round(duration * self.autoencoder.latent_token_rate)
        # content_mask: [B, L], helper_latent_mask: [B, T]
        helper_latent_mask = create_mask_from_length(n_latents.sum(1)).to(
            content_mask.device
        )
        attn_mask = ta_content_mask.unsqueeze(
            -1
        ) * helper_latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        align_path = create_alignment_path(n_latents, attn_mask)
        time_aligned_content = content[:, :trunc_ta_length]
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2).to(content.dtype), time_aligned_content
        )  # (B, T, L) x (B, L, E) -> (B, T, E)

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
        context_mask = content_mask.detach().clone()
        context_mask[is_time_aligned, 1:] = False

        # truncate dummy non time aligned context
        if is_time_aligned.sum().item() < batch_size:
            trunc_nta_length = content_mask[~is_time_aligned].sum(1).max()
        else:
            trunc_nta_length = content.size(1)
        context = context[:, :trunc_nta_length]
        context_mask = context_mask[:, :trunc_nta_length]

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
            "local_duration_loss": local_duration_loss,
            "global_duration_loss": global_duration_loss
        }

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        is_time_aligned: list[bool],
        instruction: torch.Tensor,
        instruction_lengths: Sequence[int],
        scheduler: SchedulerMixin,
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
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
        instruction_mask = create_mask_from_length(instruction_lengths)
        content, content_mask, global_duration_pred, local_duration_pred = \
            self.content_adapter(content, content_mask, instruction, instruction_mask)

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps
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
        ) - self.duration_offset  # frame number in `self.frame_resolution`
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
        context_mask[
            is_time_aligned,
            1:] = False  # only use the first dummy non time aligned embedding
        # truncate dummy non time aligned context
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
            # expand the latent if we are doing classifier free guidance
            if classifier_free_guidance:
                latent_input = torch.cat([latent, latent])
            else:
                latent_input = latent

            latent_input = scheduler.scale_model_input(latent_input, timestep)
            noise_pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=timestep,
                time_aligned_context=time_aligned_content,
                context=context,
                context_mask=context_mask
            )

            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                if guidance_rescale != 0.0:
                    noise_pred = self.rescale_cfg(
                        noise_pred_cond, noise_pred, guidance_rescale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i + 1) % scheduler.order == 0):
                progress_bar.update(1)

        progress_bar.close()

        # TODO variable length decoding, using `latent_mask`
        waveform = self.autoencoder.decode(latent)
        return waveform


class DoubleContentAudioDiffusion(CrossAttentionAudioDiffusion):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: nn.Module,
        backbone: nn.Module,
        content_dim: int,
        frame_resolution: float,
        duration_offset: float = 1.0,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        cfg_drop_ratio: float = 0.2,
    ):
        super().__init__(
            autoencoder=autoencoder,
            content_encoder=content_encoder,
            content_adapter=content_adapter,
            backbone=backbone,
            duration_offset=duration_offset,
            noise_scheduler_name=noise_scheduler_name,
            snr_gamma=snr_gamma,
            cfg_drop_ratio=cfg_drop_ratio
        )
        self.frame_resolution = frame_resolution

    def forward(
        self, content, duration, task, is_time_aligned, waveform,
        waveform_lengths, instruction, instruction_lengths, **kwargs
    ):
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

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
        context_mask = content_mask.detach()
        instruction_mask = create_mask_from_length(instruction_lengths)

        content, content_mask, global_duration_pred, local_duration_pred = \
            self.content_adapter(content, content_mask, instruction, instruction_mask)

        # TODO if all non time aligned, content length > duration length

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
        # local duration loss
        local_duration_pred = local_duration_pred[:, :trunc_ta_length]
        ta_content_mask = content_mask[:, :trunc_ta_length]
        local_duration_target = local_duration_target.to(
            dtype=local_duration_pred.dtype
        )
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

        # global duration loss
        global_duration_loss = F.mse_loss(
            global_duration_target, global_duration_pred
        )
        # --------------------------------------------------------------------
        # prepare latent and diffusion-related noise
        # --------------------------------------------------------------------
        batch_size = latent.shape[0]
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)

        # --------------------------------------------------------------------
        # duration adapter
        # --------------------------------------------------------------------
        # content_mask: [B, L], helper_latent_mask: [B, T]
        if is_time_aligned.sum() == 0 and \
            duration.size(1) < content_mask.size(1):
            # for non time-aligned tasks like TTA, `duration` is dummy one
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
        time_aligned_content = content[:, :trunc_ta_length]
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2).to(content.dtype), time_aligned_content
        )

        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        time_aligned_content = trim_or_pad_length(
            time_aligned_content, latent_length, 1
        )
        length_aligned_content = trim_or_pad_length(
            length_aligned_content, latent_length, 1
        )
        time_aligned_content = time_aligned_content + length_aligned_content
        context = content
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
            timesteps=timesteps,
            time_aligned_context=time_aligned_content,
            context=context,
            x_mask=latent_mask,
            context_mask=context_mask,
        )
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = self.loss_with_snr(pred, target, timesteps, latent_mask)
        return {
            "diff_loss": diff_loss,
            "local_duration_loss": local_duration_loss,
            "global_duration_loss": global_duration_loss,
        }

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        is_time_aligned: list[bool],
        instruction: torch.Tensor,
        instruction_lengths: Sequence[int],
        scheduler: SchedulerMixin,
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
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
        instruction_mask = create_mask_from_length(instruction_lengths)

        content, content_mask, global_duration_pred, local_duration_pred = \
            self.content_adapter(content, content_mask, instruction, instruction_mask)

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps
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
        ) - self.duration_offset  # frame number in `self.frame_resolution`
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

        # time_aligned_content[~is_time_aligned] = self.dummy_ta_embed.to(
        #     time_aligned_content.dtype
        # )

        length_aligned_content = trim_or_pad_length(
            length_aligned_content, time_aligned_content.size(1), 1
        )
        time_aligned_content = time_aligned_content + length_aligned_content

        # --------------------------------------------------------------------
        # prepare unconditional input
        # --------------------------------------------------------------------
        context = content
        # context[is_time_aligned] = self.dummy_nta_embed.to(context.dtype)
        context_mask = content_mask
        # context_mask[
        #     is_time_aligned,
        #     1:] = False  # only use the first dummy non time aligned embedding
        # # truncate dummy non time aligned context
        # if is_time_aligned.sum().item() < batch_size:
        #     trunc_nta_length = content_mask[~is_time_aligned].sum(1).max()
        # else:
        #     trunc_nta_length = content.size(1)
        # context = context[:, :trunc_nta_length]
        # context_mask = context_mask[:, :trunc_nta_length]

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
            # expand the latent if we are doing classifier free guidance
            if classifier_free_guidance:
                latent_input = torch.cat([latent, latent])
            else:
                latent_input = latent

            latent_input = scheduler.scale_model_input(latent_input, timestep)
            noise_pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask,
                timesteps=timestep,
                time_aligned_context=time_aligned_content,
                context=context,
                context_mask=context_mask
            )

            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                if guidance_rescale != 0.0:
                    noise_pred = self.rescale_cfg(
                        noise_pred_cond, noise_pred, guidance_rescale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i + 1) % scheduler.order == 0):
                progress_bar.update(1)

        progress_bar.close()

        # TODO variable length decoding, using `latent_mask`
        waveform = self.autoencoder.decode(latent)
        return waveform
