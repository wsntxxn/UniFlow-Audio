import logging
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .modules import (
    film_modulate,
    unpatchify,
    PatchEmbed,
    PE_wrapper,
    TimestepEmbedder,
    FeedForward,
    RMSNorm,
)
from .span_mask import compute_mask_indices
from .attention import Attention

logger = logging.Logger(__file__)


class AdaLN(nn.Module):
    def __init__(self, dim, ada_mode='ada', r=None, alpha=None):
        super().__init__()
        self.ada_mode = ada_mode
        self.scale_shift_table = None
        if ada_mode == 'ada':
            # move nn.silu outside
            self.time_ada = nn.Linear(dim, 6 * dim, bias=True)
        elif ada_mode == 'ada_single':
            # adaln used in pixel-art alpha
            self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
        elif ada_mode in ['ada_solo', 'ada_sola_bias']:
            self.lora_a = nn.Linear(dim, r * 6, bias=False)
            self.lora_b = nn.Linear(r * 6, dim * 6, bias=False)
            self.scaling = alpha / r
            if ada_mode == 'ada_sola_bias':
                # take bias out for consistency
                self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
        else:
            raise NotImplementedError

    def forward(self, time_token=None, time_ada=None):
        if self.ada_mode == 'ada':
            assert time_ada is None
            B = time_token.shape[0]
            time_ada = self.time_ada(time_token).reshape(B, 6, -1)
        elif self.ada_mode == 'ada_single':
            B = time_ada.shape[0]
            time_ada = time_ada.reshape(B, 6, -1)
            time_ada = self.scale_shift_table[None] + time_ada
        elif self.ada_mode in ['ada_sola', 'ada_sola_bias']:
            B = time_ada.shape[0]
            time_ada_lora = self.lora_b(self.lora_a(time_token)) * self.scaling
            time_ada = time_ada + time_ada_lora
            time_ada = time_ada.reshape(B, 6, -1)
            if self.scale_shift_table is not None:
                time_ada = self.scale_shift_table[None] + time_ada
        else:
            raise NotImplementedError
        return time_ada


class DiTBlock(nn.Module):
    """
    A modified PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """
    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        qk_norm=None,
        act_layer='gelu',
        norm_layer=nn.LayerNorm,
        time_fusion='none',
        ada_sola_rank=None,
        ada_sola_alpha=None,
        skip=False,
        skip_norm=False,
        rope_mode='none',
        context_norm=False,
        use_checkpoint=False
    ):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            rope_mode=rope_mode
        )

        if context_dim is not None:
            self.use_context = True
            self.cross_attn = Attention(
                dim=dim,
                num_heads=num_heads,
                context_dim=context_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                qk_norm=qk_norm,
                rope_mode='none'
            )
            self.norm2 = norm_layer(dim)
            if context_norm:
                self.norm_context = norm_layer(context_dim)
            else:
                self.norm_context = nn.Identity()
        else:
            self.use_context = False

        self.norm3 = norm_layer(dim)
        self.mlp = FeedForward(
            dim=dim, mult=mlp_ratio, activation_fn=act_layer, dropout=0
        )

        self.use_adanorm = True if time_fusion != 'token' else False
        if self.use_adanorm:
            self.adaln = AdaLN(
                dim,
                ada_mode=time_fusion,
                r=ada_sola_rank,
                alpha=ada_sola_alpha
            )
        if skip:
            self.skip_norm = norm_layer(2 *
                                        dim) if skip_norm else nn.Identity()
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        x,
        time_token=None,
        time_ada=None,
        skip=None,
        context=None,
        x_mask=None,
        context_mask=None,
        extras=None
    ):
        if self.use_checkpoint:
            return checkpoint(
                self._forward,
                x,
                time_token,
                time_ada,
                skip,
                context,
                x_mask,
                context_mask,
                extras,
                use_reentrant=False
            )
        else:
            return self._forward(
                x, time_token, time_ada, skip, context, x_mask, context_mask,
                extras
            )

    def _forward(
        self,
        x,
        time_token=None,
        time_ada=None,
        skip=None,
        context=None,
        x_mask=None,
        context_mask=None,
        extras=None
    ):
        B, T, C = x.shape
        if self.skip_linear is not None:
            assert skip is not None
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        if self.use_adanorm:
            time_ada = self.adaln(time_token, time_ada)
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = time_ada.chunk(6, dim=1)

        # self attention
        if self.use_adanorm:
            x_norm = film_modulate(
                self.norm1(x), shift=shift_msa, scale=scale_msa
            )
            x = x + (1-gate_msa) * self.attn(
                x_norm, context=None, context_mask=x_mask, extras=extras
            )
        else:
            x = x + self.attn(
                self.norm1(x),
                context=None,
                context_mask=x_mask,
                extras=extras
            )

        # cross attention
        if self.use_context:
            assert context is not None
            x = x + self.cross_attn(
                x=self.norm2(x),
                context=self.norm_context(context),
                context_mask=context_mask,
                extras=extras
            )

        # mlp
        if self.use_adanorm:
            x_norm = film_modulate(
                self.norm3(x), shift=shift_mlp, scale=scale_mlp
            )
            x = x + (1-gate_mlp) * self.mlp(x_norm)
        else:
            x = x + self.mlp(self.norm3(x))

        return x


class FinalBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        patch_size,
        in_chans,
        img_size,
        input_type='2d',
        norm_layer=nn.LayerNorm,
        use_conv=True,
        use_adanorm=True
    ):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.input_type = input_type

        self.norm = norm_layer(embed_dim)
        if use_adanorm:
            self.use_adanorm = True
        else:
            self.use_adanorm = False

        if input_type == '2d':
            self.patch_dim = patch_size**2 * in_chans
            self.linear = nn.Linear(embed_dim, self.patch_dim, bias=True)
            if use_conv:
                self.final_layer = nn.Conv2d(
                    self.in_chans, self.in_chans, 3, padding=1
                )
            else:
                self.final_layer = nn.Identity()

        elif input_type == '1d':
            self.patch_dim = patch_size * in_chans
            self.linear = nn.Linear(embed_dim, self.patch_dim, bias=True)
            if use_conv:
                self.final_layer = nn.Conv1d(
                    self.in_chans, self.in_chans, 3, padding=1
                )
            else:
                self.final_layer = nn.Identity()

    def forward(self, x, time_ada=None, extras=0):
        B, T, C = x.shape
        x = x[:, extras:, :]
        # only handle generation target
        if self.use_adanorm:
            shift, scale = time_ada.reshape(B, 2, -1).chunk(2, dim=1)
            x = film_modulate(self.norm(x), shift, scale)
        else:
            x = self.norm(x)
        x = self.linear(x)
        x = unpatchify(x, self.in_chans, self.input_type, self.img_size)
        x = self.final_layer(x)
        return x


class UDiT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        input_type='2d',
        out_chans=None,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        qk_norm=None,
        act_layer='gelu',
        norm_layer='layernorm',
        context_norm=False,
        use_checkpoint=False,
        # time fusion ada or token
        time_fusion='token',
        ada_sola_rank=None,
        ada_sola_alpha=None,
        cls_dim=None,
        # max length is only used for concat
        context_dim=768,
        context_fusion='concat',
        context_max_length=128,
        context_pe_method='sinu',
        pe_method='abs',
        rope_mode='none',
        use_conv=True,
        skip=True,
        skip_norm=True
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # input
        self.in_chans = in_chans
        self.input_type = input_type
        if self.input_type == '2d':
            num_patches = (img_size[0] //
                           patch_size) * (img_size[1] // patch_size)
        elif self.input_type == '1d':
            num_patches = img_size // patch_size
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            input_type=input_type
        )
        out_chans = in_chans if out_chans is None else out_chans
        self.out_chans = out_chans

        # position embedding
        self.rope = rope_mode
        self.x_pe = PE_wrapper(
            dim=embed_dim, method=pe_method, length=num_patches
        )

        logger.info(f'x position embedding: {pe_method}')
        logger.info(f'rope mode: {self.rope}')

        # time embed
        self.time_embed = TimestepEmbedder(embed_dim)
        self.time_fusion = time_fusion
        self.use_adanorm = False

        # cls embed
        if cls_dim is not None:
            self.cls_embed = nn.Sequential(
                nn.Linear(cls_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),
            )
        else:
            self.cls_embed = None

        # time fusion
        if time_fusion == 'token':
            # put token at the beginning of sequence
            self.extras = 2 if self.cls_embed else 1
            self.time_pe = PE_wrapper(
                dim=embed_dim, method='abs', length=self.extras
            )
        elif time_fusion in ['ada', 'ada_single', 'ada_sola', 'ada_sola_bias']:
            self.use_adanorm = True
            # aviod  repetitive silu for each adaln block
            self.time_act = nn.SiLU()
            self.extras = 0
            self.time_ada_final = nn.Linear(
                embed_dim, 2 * embed_dim, bias=True
            )
            if time_fusion in ['ada_single', 'ada_sola', 'ada_sola_bias']:
                # shared adaln
                self.time_ada = nn.Linear(embed_dim, 6 * embed_dim, bias=True)
            else:
                self.time_ada = None
        else:
            raise NotImplementedError
        logger.info(f'time fusion mode: {self.time_fusion}')

        # context
        # use a simple projection
        self.use_context = False
        self.context_cross = False
        self.context_max_length = context_max_length
        self.context_fusion = 'none'
        if context_dim is not None:
            self.use_context = True
            self.context_embed = nn.Sequential(
                nn.Linear(context_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),
            )
            self.context_fusion = context_fusion
            if context_fusion == 'concat' or context_fusion == 'joint':
                self.extras += context_max_length
                self.context_pe = PE_wrapper(
                    dim=embed_dim,
                    method=context_pe_method,
                    length=context_max_length
                )
                # no cross attention layers
                context_dim = None
            elif context_fusion == 'cross':
                self.context_pe = PE_wrapper(
                    dim=embed_dim,
                    method=context_pe_method,
                    length=context_max_length
                )
                self.context_cross = True
                context_dim = embed_dim
            else:
                raise NotImplementedError
        logger.info(f'context fusion mode: {context_fusion}')
        logger.info(f'context position embedding: {context_pe_method}')

        self.use_skip = skip

        # norm layers
        if norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'rmsnorm':
            norm_layer = RMSNorm
        else:
            raise NotImplementedError

        logger.info(f'use long skip connection: {skip}')
        self.in_blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                qk_norm=qk_norm,
                act_layer=act_layer,
                norm_layer=norm_layer,
                time_fusion=time_fusion,
                ada_sola_rank=ada_sola_rank,
                ada_sola_alpha=ada_sola_alpha,
                skip=False,
                skip_norm=False,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint
            ) for _ in range(depth // 2)
        ])

        self.mid_block = DiTBlock(
            dim=embed_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            act_layer=act_layer,
            norm_layer=norm_layer,
            time_fusion=time_fusion,
            ada_sola_rank=ada_sola_rank,
            ada_sola_alpha=ada_sola_alpha,
            skip=False,
            skip_norm=False,
            rope_mode=self.rope,
            context_norm=context_norm,
            use_checkpoint=use_checkpoint
        )

        self.out_blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                qk_norm=qk_norm,
                act_layer=act_layer,
                norm_layer=norm_layer,
                time_fusion=time_fusion,
                ada_sola_rank=ada_sola_rank,
                ada_sola_alpha=ada_sola_alpha,
                skip=skip,
                skip_norm=skip_norm,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint
            ) for _ in range(depth // 2)
        ])

        # FinalLayer block
        self.use_conv = use_conv
        self.final_block = FinalBlock(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            in_chans=out_chans,
            input_type=input_type,
            norm_layer=norm_layer,
            use_conv=use_conv,
            use_adanorm=self.use_adanorm
        )
        self.initialize_weights()

    def _init_ada(self):
        if self.time_fusion == 'ada':
            nn.init.constant_(self.time_ada_final.weight, 0)
            nn.init.constant_(self.time_ada_final.bias, 0)
            for block in self.in_blocks:
                nn.init.constant_(block.adaln.time_ada.weight, 0)
                nn.init.constant_(block.adaln.time_ada.bias, 0)
            nn.init.constant_(self.mid_block.adaln.time_ada.weight, 0)
            nn.init.constant_(self.mid_block.adaln.time_ada.bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.adaln.time_ada.weight, 0)
                nn.init.constant_(block.adaln.time_ada.bias, 0)
        elif self.time_fusion == 'ada_single':
            nn.init.constant_(self.time_ada.weight, 0)
            nn.init.constant_(self.time_ada.bias, 0)
            nn.init.constant_(self.time_ada_final.weight, 0)
            nn.init.constant_(self.time_ada_final.bias, 0)
        elif self.time_fusion in ['ada_sola', 'ada_sola_bias']:
            nn.init.constant_(self.time_ada.weight, 0)
            nn.init.constant_(self.time_ada.bias, 0)
            nn.init.constant_(self.time_ada_final.weight, 0)
            nn.init.constant_(self.time_ada_final.bias, 0)
            for block in self.in_blocks:
                nn.init.kaiming_uniform_(
                    block.adaln.lora_a.weight, a=math.sqrt(5)
                )
                nn.init.constant_(block.adaln.lora_b.weight, 0)
            nn.init.kaiming_uniform_(
                self.mid_block.adaln.lora_a.weight, a=math.sqrt(5)
            )
            nn.init.constant_(self.mid_block.adaln.lora_b.weight, 0)
            for block in self.out_blocks:
                nn.init.kaiming_uniform_(
                    block.adaln.lora_a.weight, a=math.sqrt(5)
                )
                nn.init.constant_(block.adaln.lora_b.weight, 0)

    def initialize_weights(self):
        # Basic init for all layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # init patch Conv like Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Zero-out AdaLN
        if self.use_adanorm:
            self._init_ada()

        # Zero-out Cross Attention
        if self.context_cross:
            for block in self.in_blocks:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)
            nn.init.constant_(self.mid_block.cross_attn.proj.weight, 0)
            nn.init.constant_(self.mid_block.cross_attn.proj.bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out cls embedding
        if self.cls_embed:
            if self.use_adanorm:
                nn.init.constant_(self.cls_embed[-1].weight, 0)
                nn.init.constant_(self.cls_embed[-1].bias, 0)

        # Zero-out Output
        # might not zero-out this when using v-prediction
        # it could be good when using noise-prediction
        # nn.init.constant_(self.final_block.linear.weight, 0)
        # nn.init.constant_(self.final_block.linear.bias, 0)
        # if self.use_conv:
        #     nn.init.constant_(self.final_block.final_layer.weight.data, 0)
        #     nn.init.constant_(self.final_block.final_layer.bias, 0)

        # init out Conv
        if self.use_conv:
            nn.init.xavier_uniform_(self.final_block.final_layer.weight)
            nn.init.constant_(self.final_block.final_layer.bias, 0)

    def _concat_x_context(self, x, context, x_mask=None, context_mask=None):
        assert context.shape[-2] == self.context_max_length
        # Check if either x_mask or context_mask is provided
        B = x.shape[0]
        # Create default masks if they are not provided
        if x_mask is None:
            x_mask = torch.ones(B, x.shape[-2], device=x.device).bool()
        if context_mask is None:
            context_mask = torch.ones(
                B, context.shape[-2], device=context.device
            ).bool()
        # Concatenate the masks along the second dimension (dim=1)
        x_mask = torch.cat([context_mask, x_mask], dim=1)
        # Concatenate context and x along the second dimension (dim=1)
        x = torch.cat((context, x), dim=1)
        return x, x_mask

    def forward(
        self,
        x,
        timesteps,
        context,
        x_mask=None,
        context_mask=None,
        cls_token=None,
        controlnet_skips=None,
    ):
        # make it compatible with int time step during inference
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0]
                                        ).to(x.device, dtype=torch.long)

        x = self.patch_embed(x)
        x = self.x_pe(x)

        B, L, D = x.shape

        if self.use_context:
            context_token = self.context_embed(context)
            context_token = self.context_pe(context_token)
            if self.context_fusion == 'concat' or self.context_fusion == 'joint':
                x, x_mask = self._concat_x_context(
                    x=x,
                    context=context_token,
                    x_mask=x_mask,
                    context_mask=context_mask
                )
                context_token, context_mask = None, None
        else:
            context_token, context_mask = None, None

        time_token = self.time_embed(timesteps)
        if self.cls_embed:
            cls_token = self.cls_embed(cls_token)
        time_ada = None
        time_ada_final = None
        if self.use_adanorm:
            if self.cls_embed:
                time_token = time_token + cls_token
            time_token = self.time_act(time_token)
            time_ada_final = self.time_ada_final(time_token)
            if self.time_ada is not None:
                time_ada = self.time_ada(time_token)
        else:
            time_token = time_token.unsqueeze(dim=1)
            if self.cls_embed:
                cls_token = cls_token.unsqueeze(dim=1)
                time_token = torch.cat([time_token, cls_token], dim=1)
            time_token = self.time_pe(time_token)
            x = torch.cat((time_token, x), dim=1)
            if x_mask is not None:
                x_mask = torch.cat([
                    torch.ones(B, time_token.shape[1],
                               device=x_mask.device).bool(), x_mask
                ],
                                   dim=1)
            time_token = None

        skips = []
        for blk in self.in_blocks:
            x = blk(
                x=x,
                time_token=time_token,
                time_ada=time_ada,
                skip=None,
                context=context_token,
                x_mask=x_mask,
                context_mask=context_mask,
                extras=self.extras
            )
            if self.use_skip:
                skips.append(x)

        x = self.mid_block(
            x=x,
            time_token=time_token,
            time_ada=time_ada,
            skip=None,
            context=context_token,
            x_mask=x_mask,
            context_mask=context_mask,
            extras=self.extras
        )
        for blk in self.out_blocks:
            if self.use_skip:
                skip = skips.pop()
                if controlnet_skips:
                    # add to skip like u-net controlnet
                    skip = skip + controlnet_skips.pop()
            else:
                skip = None
                if controlnet_skips:
                    # directly add to x
                    x = x + controlnet_skips.pop()

            x = blk(
                x=x,
                time_token=time_token,
                time_ada=time_ada,
                skip=skip,
                context=context_token,
                x_mask=x_mask,
                context_mask=context_mask,
                extras=self.extras
            )

        x = self.final_block(x, time_ada=time_ada_final, extras=self.extras)

        return x


class MaskDiT(nn.Module):
    def __init__(
        self,
        model: UDiT,
        mae=False,
        mae_prob=0.5,
        mask_ratio=[0.25, 1.0],
        mask_span=10,
    ):
        super().__init__()
        self.model = model
        self.mae = mae
        if self.mae:
            out_channel = model.out_chans
            self.mask_embed = nn.Parameter(torch.zeros((out_channel)))
            self.mae_prob = mae_prob
            self.mask_ratio = mask_ratio
            self.mask_span = mask_span

    def random_masking(self, gt, mask_ratios, mae_mask_infer=None):
        B, D, L = gt.shape
        if mae_mask_infer is None:
            # mask = torch.rand(B, L).to(gt.device) < mask_ratios.unsqueeze(1)
            mask_ratios = mask_ratios.cpu().numpy()
            mask = compute_mask_indices(
                shape=[B, L],
                padding_mask=None,
                mask_prob=mask_ratios,
                mask_length=self.mask_span,
                mask_type="static",
                mask_other=0.0,
                min_masks=1,
                no_overlap=False,
                min_space=0,
            )
            mask = mask.unsqueeze(1).expand_as(gt)
        else:
            mask = mae_mask_infer
            mask = mask.expand_as(gt)
        gt[mask] = self.mask_embed.view(1, D, 1).expand_as(gt)[mask]
        return gt, mask.type_as(gt)

    def forward(
        self,
        x,
        timesteps,
        context,
        x_mask=None,
        context_mask=None,
        cls_token=None,
        gt=None,
        mae_mask_infer=None,
        forward_model=True
    ):
        # todo: handle controlnet inside
        mae_mask = torch.ones_like(x)
        if self.mae:
            if gt is not None:
                B, D, L = gt.shape
                mask_ratios = torch.FloatTensor(B).uniform_(*self.mask_ratio
                                                           ).to(gt.device)
                gt, mae_mask = self.random_masking(
                    gt, mask_ratios, mae_mask_infer
                )
                # apply mae only to the selected batches
                if mae_mask_infer is None:
                    # determine mae batch
                    mae_batch = torch.rand(B) < self.mae_prob
                    gt[~mae_batch] = self.mask_embed.view(
                        1, D, 1
                    ).expand_as(gt)[~mae_batch]
                    mae_mask[~mae_batch] = 1.0
            else:
                B, D, L = x.shape
                gt = self.mask_embed.view(1, D, 1).expand_as(x)
            x = torch.cat([x, gt, mae_mask[:, 0:1, :]], dim=1)

        if forward_model:
            x = self.model(
                x=x,
                timesteps=timesteps,
                context=context,
                x_mask=x_mask,
                context_mask=context_mask,
                cls_token=cls_token
            )
            # logger.info(mae_mask[:, 0, :].sum(dim=-1))
        return x, mae_mask
