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

# mixin辅助类
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
        snr_gamma: float = None,# 用于信噪比（SNR，Signal-to-Noise Ratio）调整，以平衡高低频噪声的处理
        classifier_free_guidance: bool = True,
    ):
        # 显式初始化nn.Modeule,因为backbone等参数是nn.Module或其子类
        nn.Module.__init__(self)
        #继承了多个父类，因此显式调用DiffusionMixin初始化方法
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
        # ---------------------------------------------------------------------------- #
        #                                      cfg                                     #
        # ---------------------------------------------------------------------------- #
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
        # ---------------------------------------------------------------------------- #
        #                                   准备content，当启用cfg时，content长度也要变成两倍                                  #
        # ---------------------------------------------------------------------------- #
        if classifier_free_guidance:
            content, content_mask = self.encode_content_classifier_free(
                content, task, num_samples_per_content
            )
        else:
            content, content_mask = self.content_encoder.encode_content(
                content, task, device=device
            )
            # repeat_interleave是tensor方法，沿着dim=0对 content 进行复制，使每个 content 具有 num_samples_per_content 份
            content = content.repeat_interleave(num_samples_per_content, 0)
            content_mask = content_mask.repeat_interleave(
                num_samples_per_content, 0
            )

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps
        #latent shape 直接由传入参数给定，不使用预测时长
        latent = self.prepare_latent(
            batch_size, scheduler, latent_shape, content.dtype, device
        )
        #scheduler.order=1，控制去噪时使用的历史时间步数，因此这里得到的预热步数为0
        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, timestep in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            #如果启用了 classifier_free_guidance，则 复制 latent，目的是计算有引导和无引导的噪声预测
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
                #chunk(2)：将 noise_pred 分成两部分
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                #即(1-\omega)*uncond+\omega*cond,将条件预测和无条件预测线性组合
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_content-noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # 更新进度条
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
        #准备[uncond conten, cond content]
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
    def encode_content_task_classifier_free(
        self,
        content: list[Any],
        task: list[str],
        num_samples_per_content: int = 1
    ):
        #准备[uncond conten, cond content]
        device = self.dummy_param.device
        content, content_mask,task_emb = self.content_encoder.encode_task_content(
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
        task_emb=torch.cat([task_emb,task_emb])

        return content, content_mask, task_emb


class VariableLengthAudioDiffusion(AudioDiffusion):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: nn.Module,
        backbone: nn.Module,
        # latent_dim: int,
        frame_resolution:float,  # frame resolution in second for duration prediction
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        classifier_free_guidance: bool = False,
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
        ###################################################
        #准备wavform 的latent
        ###################################################
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        
        # 使用VAE编码波形文件得到latent和latent_mask
        #latent:(B,E,L) latent_mask:(B,L)
        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        # content: (B, T, E) content_mask: (B, T)
        # ---------------------------------------------------------------------------- #
        #                   content encoder              #
        # ---------------------------------------------------------------------------- #
        content, content_mask = self.content_encoder.encode_content(
            content, task, device=device
        )
        # ---------------------------------------------------------------------------- #
        #                                  add spk emb                                 #
        # ---------------------------------------------------------------------------- #
        
        
        
        
         # ---------------------------------------------------------------------------- #
         #                          content_adapter对content进行编码                         #
         # ---------------------------------------------------------------------------- #
     
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,
        ) = self.content_adapter(content, content_mask)
        
        # duration = torch.round(duration * self.autoencoder.latent_token_rate)
        # 获得duration的gt
        n_frames = torch.round(duration / self.frame_resolution)
        local_duration_target = torch.log(n_frames + 1e-8)
        global_duration_target = torch.log(
            latent_mask.sum(1) / self.autoencoder.latent_token_rate + 1e-8
        )
        # 计算duraion的loss
        local_duration_loss = loss_with_mask(
            (local_duration_target - local_duration_pred)**2, content_mask
        )
        global_duration_loss = torch.sum(
            (global_duration_target - global_duration_pred)**2
        ) / content.shape[0]

        # --------------------------------------------------------------------
        # CFG 
        # --------------------------------------------------------------------
        #如果启用CFG，则对content进行随机mask，概率为0.1
        #此处不启用CFG
        # 在部分训练步骤中 去掉条件信息，让模型学会如何 不依赖输入条件也能生成高质量结果，从而增强生成多样性。
        if self.training and self.classifier_free_guidance:
            #len() 作用于 waveform 这样的 二维张量（形状 [96, 241195]），它会返回 第一维的大小，也就是 批量大小（batch_size）。
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < 0.1
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0
        # ---------------------------------------------------------------------------- #
        #                                 获取gt噪声样本和预测噪声目标                                #
        # ---------------------------------------------------------------------------- #
        batch_size = latent.shape[0]
        #获取随机的加噪步数序列
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        #加噪
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        #获取预测目标，噪声本身或者是速度预测
        target = self.get_target(latent, noise, timesteps)

        # --------------------------------------------------------------------
        # duration adapter，对音素content做帧拓展.完全看不懂😩
        #为什么使用duration，不使用预测出来的local_duration_pred？
        # --------------------------------------------------------------------

        # content_mask: [B, L], helper_latent_mask: [B, T]，n_latents为转换为token的local duraion
        n_latents = torch.round(duration * self.autoencoder.latent_token_rate)
        helper_latent_mask = create_mask_from_length(n_latents.sum(1)).to(
            content_mask.device
        )
        
    #     	•	content_mask: [B, L] → content_mask.unsqueeze(-1): [B, L, 1]
	# •	helper_latent_mask: [B, T] → helper_latent_mask.unsqueeze(1): [B, 1, T]
	# •	通过 广播机制，两者相乘，得到 [B, L, T] 的注意力掩码矩阵：
	# •	1 * 1 = 1（可用位置）。
	# •	1 * 0 = 0 或 0 * 1 = 0（屏蔽位置）。
        attn_mask = content_mask.unsqueeze(-1
                                          ) * helper_latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        # align_path: [B, L, T] 
        align_path = create_alignment_path(n_latents, attn_mask)
        
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2), content
        )  # 矩阵相乘(B, T, L) x (B, L, E) -> (B, T, E)

        # --------------------------------------------------------------------
    
        #将time_aligned_content 填充或截断到与noisy_latent相同的长度
        #latent noise形状直接由waveform latent得到，因为这是在准备输入到backbone的latent+noise
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
        
        # ---------------------------------------------------------------------------- #
        #                                 准备上下文context                                 #
        # ---------------------------------------------------------------------------- #
        #dummy_nta_embed 初始化为d_out维的可训练的0向量
        #nta为Non-Temporal Alignment（非时间对齐）的缩写
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
        #例如num_steps=20，,则降噪timesteps为[951,901,...,51,1]，共20个
        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps
        local_duration_pred = torch.exp(local_duration_pred) * content_mask
        #torch.ceil(local_duration_pred) 计算向上取整，确保时长是整数（帧数不能是小数）。
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
        # duration adapter 帧拓展
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
        # 使用预测的时长作为latent的长度
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
        #scheduler.order=1，控制去噪时使用的历史时间步数，因此这里得到的预热步数为0
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
            #dummy_nta embed形状为[256]
            # (None, ) * (content.ndim - 1) 生成 (None, None, ..., None)，用于扩展 dummy_nta_embed 的维度，使其能够匹配 content。
	        # 代表 保持 dummy_nta_embed 的最后一个维度，即不变。
            #.expand_as(content) 将 context 扩展为与 content 相同的形状，但不复制数据，只是改变视图（view）。
            #最终dummy_nta embed为[1,L,256]
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
        
        # ---------------------------------------------------------------------------- #
        #           相较于VariableLengthAudioDiffusion，多了一个time_aligned_content2          #
        # ---------------------------------------------------------------------------- #

        time_aligned_content2 = self.content_encoder.encode_time_aligned_content(
            content_input, task, device=device
        )

        # --------------------------------------------------------------------
        # prepare input to the backbone
        #将time_aligned_content1和time_aligned_content2填充或截断到与noisy_latent相同的长度
        #再将time_aligned_content1和time_aligned_content2叠加拼接
        # --------------------------------------------------------------------
        # TODO compatility for 2D spectrogram VAE
        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        print(f'time_aligned_content1 before trim_or_pad_length: {time_aligned_content1.size()}')
        time_aligned_content1 = trim_or_pad_length(
            time_aligned_content1, latent_length, 1
        )
        print(f'after trim_or_pad_length: {time_aligned_content1.size()}')
        print(f'time_aligned_content2 before trim_or_pad_length: {time_aligned_content2.size()}')
        time_aligned_content2 = trim_or_pad_length(
            time_aligned_content2, latent_length, 1
        )
        print(f'time_aligned_content2 after trim_or_pad_length: {time_aligned_content2.size()}')
        # time_aligned_content1: from unaligned input (phoneme)
        # time_aligned_content2: from aligned input (f0/energy)，midi？
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


class MultiTaskDiffusion(AudioDiffusion):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        content_adapter: nn.Module,
        backbone: nn.Module,
        # latent_dim: int,
        frame_resolution:float,  # frame resolution in second for duration prediction
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
        ###################################################
        #准备wavform 的latent
        ###################################################
        device = self.dummy_param.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        
        # 使用VAE编码波形文件得到latent和latent_mask
        #latent:(B,E,L) latent_mask:(B,L)
        #wave length是frame数=secs*sr
        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        # content: (B, T, E) content_mask: (B, T)
        # ---------------------------------------------------------------------------- #
        #                   content encoder ,编码task             #
        #task_emb=[B,1,d_model]
        # ---------------------------------------------------------------------------- #
        content, content_mask,task_emb = self.content_encoder.encode_task_content(
            content, task, device=device
        )
        # tta text经过embed之后可能比音素的最大长度长，统一content、content_mask和duration形状
        if content.shape[1] != duration.shape[1]:
            print(f'before_pad:content.shape{content.shape},  duration.shape{duration.shape}')
            duration=F.pad(duration, (0, content.shape[1]-duration.shape[1]))
            print(f'after_pad:content.shape{content.shape},  duration.shape{duration.shape}')


         # ---------------------------------------------------------------------------- #
         #                  content_adapter对content进行编码。时长预测，加入task emb                      #
         # ---------------------------------------------------------------------------- #
     
        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,
        ) = self.content_adapter(content, content_mask,task_emb)
        print(f'\nglobal_duration_pred.shape{global_duration_pred}')
        
        # #tta text经过embed之后可能比音素的最大长度长，因此需要将duration为pred的形状
        # if duration.shape != local_duration_pred.shape:
        #     print(f'before_pad:duration.shape{duration.shape},  local_duration_pred.shape{local_duration_pred.shape}')
        #     duration=F.pad(duration, (0, local_duration_pred.shape[1]-duration.shape[1]))
        #     print(f'after_pad:duration.shape{duration.shape},  local_duration_pred.shape{local_duration_pred.shape}')

        # duration = torch.round(duration * self.autoencoder.latent_token_rate)
        # 获得duration的gt
        n_frames = torch.round(duration / self.frame_resolution)
        local_duration_target = torch.log(n_frames + 1e-8)
        # ---------------------------------------------------------------------------- #
        #                          global_duration_target有问题？
        #  latent_mask.sum(1) / self.autoencoder.latent_token_rate 算出秒数，秒数取log                    #
        # ---------------------------------------------------------------------------- #
        global_duration_target = torch.log(
            latent_mask.sum(1) / self.autoencoder.latent_token_rate + 1e-8
        )
        # 计算duraion的loss
        local_duration_loss = loss_with_mask(
            (local_duration_target - local_duration_pred)**2, content_mask
        )    
        # why？
        global_duration_loss = torch.sum(
            (global_duration_target - global_duration_pred)**2
        ) / content.shape[0]

        # --------------------------------------------------------------------
        # CFG 
        # --------------------------------------------------------------------
        #如果启用CFG，则对content进行随机mask，概率为0.1
        #此处不启用CFG
        # 在部分训练步骤中 去掉条件信息，让模型学会如何 不依赖输入条件也能生成高质量结果，从而增强生成多样性。
        if self.training and self.classifier_free_guidance:
            #len() 作用于 waveform 这样的 二维张量（形状 [96, 241195]），它会返回 第一维的大小，也就是 批量大小（batch_size）。
            mask_indices = [
                k for k in range(len(waveform)) if random.random() < 0.1
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0
        # ---------------------------------------------------------------------------- #
        #                                 获取gt噪声样本和预测噪声目标                                #
        # ---------------------------------------------------------------------------- #
        batch_size = latent.shape[0]
        #获取随机的加噪步数序列
        timesteps = self.get_timesteps(batch_size, device, self.training)
        noise = torch.randn_like(latent)
        #加噪
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        #获取预测目标，噪声本身或者是速度预测
        target = self.get_target(latent, noise, timesteps)

        # --------------------------------------------------------------------
        # duration adapter，对音素content做帧拓展.完全看不懂😩
        #为什么使用duration，不使用预测出来的local_duration_pred？
        # --------------------------------------------------------------------

        # content_mask: [B, L], helper_latent_mask: [B, T]
        n_latents = torch.round(duration * self.autoencoder.latent_token_rate)
        helper_latent_mask = create_mask_from_length(n_latents.sum(1)).to(
            content_mask.device
        )
        
    #     	•	content_mask: [B, L] → content_mask.unsqueeze(-1): [B, L, 1]
	# •	helper_latent_mask: [B, T] → helper_latent_mask.unsqueeze(1): [B, 1, T]
	# •	通过 广播机制，两者相乘，得到 [B, L, T] 的注意力掩码矩阵：
	# •	1 * 1 = 1（可用位置）。
	# •	1 * 0 = 0 或 0 * 1 = 0（屏蔽位置）。
        attn_mask = content_mask.unsqueeze(-1
                                          ) * helper_latent_mask.unsqueeze(1)
        # attn_mask: [B, L, T]
        # align_path: [B, L, T] 
        align_path = create_alignment_path(n_latents, attn_mask)
        
        time_aligned_content = torch.matmul(
            align_path.transpose(1, 2), content
        )  # 矩阵相乘(B, T, L) x (B, L, E) -> (B, T, E)

        # --------------------------------------------------------------------
    
        #将time_aligned_content 填充或截断到与noisy_latent相同的长度
        #latent noise形状直接由waveform latent得到，因为这是在准备输入到backbone的latent+noise
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
        
        # ---------------------------------------------------------------------------- #
        #                                 准备上下文context                                 #
        # ---------------------------------------------------------------------------- #
        #dummy_nta_embed 初始化为d_out维的可训练的0向量
        #nta为Non-Temporal Alignment（非时间对齐）的缩写
        # context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
        #                                (..., )].expand_as(content)
        # context_mask = torch.ones(batch_size, context.size(1)).to(device)
        
        pred: torch.Tensor = self.backbone(
            # x=latent_input,
            x=noisy_latent,
            timesteps=timesteps,
            time_aligned_context=time_aligned_content,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
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
        scheduler: SchedulerMixin,
        num_steps: int = 20,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        num_samples_per_content: int = 1,
        use_gt_duration: bool = False,
        **kwargs
    ):
        device = self.dummy_param.device

        classifier_free_guidance = guidance_scale > 1.0   
        
        
        # classifier_free_guidance=False


        batch_size = len(content) * num_samples_per_content
        # ---------------------------------------------------------------------------- #
        #                                   准备content，当启用cfg时，content长度也要变成两倍 [uncond_content,cond_content]                                 #
        # ---------------------------------------------------------------------------- #
        # if classifier_free_guidance:
        #     content, content_mask,task_emb = self.encode_content_task_classifier_free(
        #         content, task, num_samples_per_content
        #     )
        # else:
        #     content, content_mask,task_emb = self.content_encoder.encode_task_content(
        #         content, task, device=device
        #     )
        #     # repeat_interleave是tensor方法，沿着dim=0对 content 进行复制，使每个 content 具有 num_samples_per_content 份
        #     content = content.repeat_interleave(num_samples_per_content, 0)
        #     content_mask = content_mask.repeat_interleave(
        #         num_samples_per_content, 0
        #     )

        content, content_mask,task_emb = self.content_encoder.encode_task_content(
            content, task, device=device
        )
        # repeat_interleave是tensor方法，沿着dim=0对 content 进行复制，使每个 content 具有 num_samples_per_content 份
        content = content.repeat_interleave(num_samples_per_content, 0)
        content_mask = content_mask.repeat_interleave(
            num_samples_per_content, 0
        )


        (
            content,
            content_mask,
            global_duration_pred,
            local_duration_pred,  # log(n_frames)
        ) = self.content_adapter(content, content_mask,task_emb)

        #CFG
        if classifier_free_guidance:
            uncond_content = torch.zeros_like(content)
            uncond_content_mask = content_mask.detach().clone()
            content = torch.cat([uncond_content, content])
            content_mask = torch.cat([uncond_content_mask, content_mask])



        
        #例如num_steps=20，,则降噪timesteps为[951,901,...,51,1]，共20个
        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        local_duration_pred = torch.exp(local_duration_pred) * content_mask
        #torch.ceil(local_duration_pred) 计算向上取整，确保时长是整数（帧数不能是小数）。
        local_duration_pred = torch.ceil(local_duration_pred)  # n_frames
        #转换为token数
        local_duration_pred = torch.round(local_duration_pred * self.frame_resolution * \
            self.autoencoder.latent_token_rate)
        #转换为秒数
        global_duration_pred=torch.exp(global_duration_pred)
        # 转换为token数
        global_duration_pred=torch.round(global_duration_pred*self.autoencoder.latent_token_rate).to(device)
        
        # TODO：此处未匹配cfg、tts任务，待补充。
        if use_gt_duration and "duration" in kwargs:
            local_duration_pred = torch.round(
                torch.as_tensor(kwargs["duration"]) *
                self.autoencoder.latent_token_rate
            ).to(device)

        # --------------------------------------------------------------------
        # duration adapter 帧拓展
        # --------------------------------------------------------------------
        # content_mask: [B, L], latent_mask: [B, T]       
        # if task[0]=="text_to_speech":                               
        #     global_duration = local_duration_pred.sum(1)
        # elif task[0]=="text_to_audio":
        #     global_duration=torch.cat([global_duration_pred[1],global_duration_pred[1]])
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

        # todo:此处应该将time_aligned_content[0]置零

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        
        # 使用预测的token数作为latent的长度
        # latent_shape = tuple(
        #     int(global_duration.max().item()) if dim is None else dim
        #     for dim in self.autoencoder.latent_shape
        # )
        dim0=self.autoencoder.latent_dim
        if task[0]=="text_to_audio":
            dim1=int(global_duration_pred[0].item())
            print("tta pred global time:",dim1/50)
            #ust gt duration
            # dim1=500
            assert dim1>0
        elif task[0]=="text_to_speech":
            dim1= int(global_duration.max().item())
            print("tts pred global time:",dim1/50)
            assert dim1>0
        else:
            raise ValueError("task type error")

        latent_shape=(dim0,dim1)
        

        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=content.dtype
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latent = latent * scheduler.init_noise_sigma
        #scheduler.order=1，控制去噪时使用的历史时间步数，因此这里得到的预热步数为0
        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)


        #修剪time_align_content 到latent 长度（tta任务可能需要修剪，tts任务的unconda content 需要修剪）
        latent_length = latent.size(self.autoencoder.time_dim)
        if time_aligned_content.size(1) > latent_length:
            time_aligned_content = time_aligned_content[:, :latent_length]
        elif time_aligned_content.size(1) < latent_length:
            pad_size = latent_length - time_aligned_content.size(1)
            padding = (0, 0, 0,
                       pad_size) + (0, 0) * (time_aligned_content.ndim - 2)
            time_aligned_content = F.pad(
                time_aligned_content, padding, mode="constant", value=0
            )
        # 对于tts，uncond content latent 也要和conda content latent 一样长


        # if task[0]=="text_to_audio":
        #     latent_mask=torch.ones(batch_size,dim1,device=device)
        # # latent_mask=torch.ones(batch_size,dim1,device=device)
        # latent_mask_input=torch.cat([latent_mask,latent_mask]) if classifier_free_guidance else latent_mask 

        # latent_mask_input=latent_mask
        latent_mask=torch.ones(batch_size,dim1,device=device)
        latent_mask_input=torch.cat([latent_mask,latent_mask]) if classifier_free_guidance else latent_mask 
        
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

            latent_input = torch.cat([latent, latent]
                                    ) if classifier_free_guidance else latent


            latent_input = scheduler.scale_model_input(latent_input, timestep)
            #dummy_nta embed形状为[256]
            # (None, ) * (content.ndim - 1) 生成 (None, None, ..., None)，用于扩展 dummy_nta_embed 的维度，使其能够匹配 content。
	        # 代表 保持 dummy_nta_embed 的最后一个维度，即不变。
            #.expand_as(content) 将 context 扩展为与 content 相同的形状，但不复制数据，只是改变视图（view）。
            #最终dummy_nta embed为[1,L,256]
            # context = self.dummy_nta_embed[(None, ) * (content.ndim - 1) +
            #                                (..., )].expand_as(content)



            # context_mask = torch.ones(batch_size, context.size(1)).to(device)

            noise_pred = self.backbone(
                x=latent_input,
                x_mask=latent_mask_input,
                timesteps=timestep,
                time_aligned_context=time_aligned_content,
                context=content,
                context_mask=content_mask
            )

            # perform guidance
            if classifier_free_guidance:
                #chunk(2)：将 noise_pred 分成两部分
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                #即(1-\omega)*uncond+\omega*cond,将条件预测和无条件预测线性组合
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


