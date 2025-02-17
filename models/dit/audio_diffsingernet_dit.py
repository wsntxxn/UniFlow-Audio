import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .mask_dit import DiTBlock, FinalBlock, UDiT
from .modules import (
    film_modulate,
    PatchEmbed,
    PE_wrapper,
    TimestepEmbedder,
    RMSNorm,
)


class AudioDiTBlock(DiTBlock):
    """
    A modified DiT block with time_aligned_context add to latent.
    """
    def __init__(
        self,
        dim,
        time_aligned_context_dim,
        dilation,
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
        super().__init__(
            dim=dim,
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
            rope_mode=rope_mode,
            context_norm=context_norm,
            use_checkpoint=use_checkpoint
        )
        # time-aligned context projection
        self.ta_context_projection = nn.Linear(
            time_aligned_context_dim, 2 * dim
        )
        self.dilated_conv = nn.Conv1d(
            dim, 2 * dim, kernel_size=3, padding=dilation, dilation=dilation
        )

    def forward(
        self,
        x,
        time_aligned_context,
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
                time_aligned_context,
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
                x,
                time_aligned_context,
                time_token,
                time_ada,
                skip,
                context,
                x_mask,
                context_mask,
                extras,
            )

    def _forward(
        self,
        x,
        time_aligned_context,
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
            # TODO diffusion timestep input is not fused here
            x = x + self.attn(
                self.norm1(x),
                context=None,
                context_mask=x_mask,
                extras=extras
            )

        # time-aligned context
        time_aligned_context = self.ta_context_projection(time_aligned_context)
        x = self.dilated_conv(x.transpose(1, 2)
                             ).transpose(1, 2) + time_aligned_context

        gate, filter = torch.chunk(x, 2, dim=-1)
        x = torch.sigmoid(gate) * torch.tanh(filter)

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


class AudioUDiT(UDiT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        input_type='2d',
        out_chans=None,
        embed_dim=768,
        depth=12,
        dilation_cycle_length=4,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        qk_norm=None,
        act_layer='gelu',
        norm_layer='layernorm',
        context_norm=False,
        use_checkpoint=False,
        time_fusion='token',
        ada_sola_rank=None,
        ada_sola_alpha=None,
        cls_dim=None,
        time_aligned_context_dim=768,
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
        nn.Module.__init__(self)
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

        self.use_skip = skip

        # norm layers
        if norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'rmsnorm':
            norm_layer = RMSNorm
        else:
            raise NotImplementedError

        self.in_blocks = nn.ModuleList([
            AudioDiTBlock(
                dim=embed_dim,
                time_aligned_context_dim=time_aligned_context_dim,
                dilation=2**(i % dilation_cycle_length),
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
            ) for i in range(depth // 2)
        ])

        self.mid_block = AudioDiTBlock(
            dim=embed_dim,
            time_aligned_context_dim=time_aligned_context_dim,
            dilation=1,
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
            AudioDiTBlock(
                dim=embed_dim,
                time_aligned_context_dim=time_aligned_context_dim,
                dilation=2**(i % dilation_cycle_length),
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
            ) for i in range(depth // 2)
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

    def forward(
        self,
        x,
        timesteps,
        time_aligned_context,
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
                time_aligned_context=time_aligned_context,
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
            time_aligned_context=time_aligned_context,
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
                time_aligned_context=time_aligned_context,
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
