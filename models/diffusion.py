import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers.schedulers as noise_schedulers

from models.content_encoder.content_encoder import ContentEncoder


class DiffusionMixin:
    def __init__(
        self,
        training_noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        inference_noise_scheduler_type: str = "DDIMScheduler",
        inference_noise_scheduler_name:
        str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = True
    ) -> None:
        self.training_noise_scheduler = noise_schedulers.DDPMScheduler.from_pretrained(
            training_noise_scheduler_name, subfolder="scheduler"
        )
        self.inference_noise_scheduler = getattr(
            noise_schedulers, inference_noise_scheduler_type
        ).from_pretrained(
            inference_noise_scheduler_name, subfolder="scheduler"
        )
        self.snr_gamma = snr_gamma
        self.classifier_free_guidance = classifier_free_guidance

    def compute_snr(self, timesteps) -> torch.Tensor:
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.training_noise_scheduler.alphas_cumprod
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
                self.training_noise_scheduler.config.num_train_timesteps,
                (batch_size, ),
                device=device
            )
        else:
            # validation on half of the total timesteps
            timesteps = (
                self.training_noise_scheduler.num_train_timesteps // 2
            ) * torch.ones((batch_size, ), dtype=torch.int64, device=device)

        timesteps = timesteps.long()
        return timesteps

    def get_target(
        self, latent: torch.Tensor, noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the target for loss depending on the prediction type
        """
        if self.training_noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.training_noise_scheduler.config.prediction_type == "v_prediction":
            target = self.training_noise_scheduler.get_velocity(
                latent, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.training_noise_scheduler.config.prediction_type}"
            )
        return target

    def loss_with_snr(
        self, pred: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if self.snr_gamma is None:
            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)],
                            dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss.mean(
                dim=list(range(1, len(loss.shape)))
            ) * mse_loss_weights
            loss = loss.mean()
        return loss


class AudioDiffusion(nn.Module, DiffusionMixin):
    def __init__(
        self,
        autoencoder: nn.Module,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        training_noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        inference_noise_scheduler_type: str = "DDIMScheduler",
        inference_noise_scheduler_name:
        str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = True,
    ):
        nn.Module.__init__(self)
        DiffusionMixin.__init__(
            self, training_noise_scheduler_name,
            inference_noise_scheduler_type, inference_noise_scheduler_name,
            snr_gamma, classifier_free_guidance
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
        num_train_timesteps = self.training_noise_scheduler.num_train_timesteps
        self.training_noise_scheduler.set_timesteps(
            num_train_timesteps, device=device
        )

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
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        noisy_latent = self.training_noise_scheduler.add_noise(
            latent, noise, timesteps
        )

        target = self.get_target(latent, noise, timesteps)

        pred = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )
        loss = self.loss_with_snr(pred, target, timesteps)

        return loss

    # @torch.no_grad()
    # def inference(
    #     self,
    #     prompt,
    #     inference_scheduler,
    #     num_steps=20,
    #     guidance_scale=3,
    #     num_samples_per_prompt=1,
    #     disable_progress=True
    # ):
    #     device = self.text_encoder.device
    #     classifier_free_guidance = guidance_scale > 1.0
    #     batch_size = len(prompt) * num_samples_per_prompt

    #     if classifier_free_guidance:
    #         prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(
    #             prompt, num_samples_per_prompt
    #         )
    #     else:
    #         prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
    #         prompt_embeds = prompt_embeds.repeat_interleave(
    #             num_samples_per_prompt, 0
    #         )
    #         boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(
    #             num_samples_per_prompt, 0
    #         )

    #     inference_scheduler.set_timesteps(num_steps, device=device)
    #     timesteps = inference_scheduler.timesteps

    #     num_channels_latents = self.unet.config.in_channels
    #     latents = self.prepare_latents(
    #         batch_size, inference_scheduler, num_channels_latents,
    #         prompt_embeds.dtype, device
    #     )

    #     num_warmup_steps = len(
    #         timesteps
    #     ) - num_steps * inference_scheduler.order
    #     progress_bar = tqdm(range(num_steps), disable=disable_progress)

    #     for i, t in enumerate(timesteps):
    #         # expand the latents if we are doing classifier free guidance
    #         latent_model_input = torch.cat(
    #             [latents] * 2
    #         ) if classifier_free_guidance else latents
    #         latent_model_input = inference_scheduler.scale_model_input(
    #             latent_model_input, t
    #         )

    #         noise_pred = self.unet(
    #             latent_model_input,
    #             t,
    #             encoder_hidden_states=prompt_embeds,
    #             encoder_attention_mask=boolean_prompt_mask
    #         ).sample

    #         # perform guidance
    #         if classifier_free_guidance:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (
    #                 noise_pred_text-noise_pred_uncond
    #             )

    #         # compute the previous noisy sample x_t -> x_t-1
    #         latents = inference_scheduler.step(
    #             noise_pred, t, latents
    #         ).prev_sample

    #         # call the callback, if provided
    #         if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
    #                                        (i+1) % inference_scheduler.order
    #                                        == 0):
    #             progress_bar.update(1)

    #     if self.set_from == "pre-trained":
    #         latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()
    #                                 ).permute(0, 3, 1, 2).contiguous()
    #     return latents

    # def prepare_latents(
    #     self, batch_size, inference_scheduler, num_channels_latents, dtype,
    #     device
    # ):
    #     shape = (batch_size, num_channels_latents, 256, 16)
    #     latents = randn_tensor(
    #         shape, generator=None, device=device, dtype=dtype
    #     )
    #     # scale the initial noise by the standard deviation required by the scheduler
    #     latents = latents * inference_scheduler.init_noise_sigma
    #     return latents

    # def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
    #     device = self.text_encoder.device
    #     batch = self.tokenizer(
    #         prompt,
    #         max_length=self.tokenizer.model_max_length,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt"
    #     )
    #     input_ids, attention_mask = batch.input_ids.to(
    #         device
    #     ), batch.attention_mask.to(device)

    #     with torch.no_grad():
    #         prompt_embeds = self.text_encoder(
    #             input_ids=input_ids, attention_mask=attention_mask
    #         )[0]

    #     prompt_embeds = prompt_embeds.repeat_interleave(
    #         num_samples_per_prompt, 0
    #     )
    #     attention_mask = attention_mask.repeat_interleave(
    #         num_samples_per_prompt, 0
    #     )

    #     # get unconditional embeddings for classifier free guidance
    #     uncond_tokens = [""] * len(prompt)

    #     max_length = prompt_embeds.shape[1]
    #     uncond_batch = self.tokenizer(
    #         uncond_tokens,
    #         max_length=max_length,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     uncond_input_ids = uncond_batch.input_ids.to(device)
    #     uncond_attention_mask = uncond_batch.attention_mask.to(device)

    #     with torch.no_grad():
    #         negative_prompt_embeds = self.text_encoder(
    #             input_ids=uncond_input_ids,
    #             attention_mask=uncond_attention_mask
    #         )[0]

    #     negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
    #         num_samples_per_prompt, 0
    #     )
    #     uncond_attention_mask = uncond_attention_mask.repeat_interleave(
    #         num_samples_per_prompt, 0
    #     )

    #     # For classifier free guidance, we need to do two forward passes.
    #     # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
    #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    #     prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
    #     boolean_prompt_mask = (prompt_mask == 1).to(device)

    #     return prompt_embeds, boolean_prompt_mask
