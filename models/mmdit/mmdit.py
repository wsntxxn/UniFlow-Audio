import torch
import torch.nn as nn

from .embeddings import TimestepEmbedder
from .layers import FinalBlock, JointBlock, MMDitSingleBlock, compute_rope_rotations
from .modules import ChannelLastConv1d, ConvMLP, MLP


class MMDiT(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        ta_context_dim: int,
        context_dim: int,
        embed_dim: int,
        depth: int,
        fused_depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        hidden_dim = embed_dim
        self.hidden_dim = embed_dim
        self.num_heads = num_heads

        self.input_proj = nn.Sequential(
            ChannelLastConv1d(
                latent_dim, hidden_dim, kernel_size=7, padding=3
            ),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

        self.nta_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

        self.ta_proj = nn.Sequential(
            ChannelLastConv1d(
                ta_context_dim, hidden_dim, kernel_size=7, padding=3
            ),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

        self.nta_global_proj = nn.Linear(hidden_dim, hidden_dim)

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        self.t_embed = TimestepEmbedder(
            hidden_dim, frequency_embedding_size=hidden_dim, max_period=1
        )

        self.joint_blocks = nn.ModuleList([
            JointBlock(
                hidden_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                pre_only=(i == depth - fused_depth - 1)
            ) for i in range(depth - fused_depth)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(
                hidden_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                kernel_size=3,
                padding=1
            ) for i in range(fused_depth)
        ])

        self.initialize_weights()
        self._latent_seq_len_cached = None

    def update_seq_lengths(self, latent_seq_len: int):
        if latent_seq_len != self._latent_seq_len_cached:
            self._latent_seq_len_cached = latent_seq_len

            base_freq = 1.0
            latent_rot = compute_rope_rotations(
                latent_seq_len,
                self.hidden_dim // self.num_heads,
                10000,
                freq_scaling=base_freq,
                device=self.device
            )

            self.latent_rot = nn.Buffer(latent_rot, persistent=False)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(
                block.latent_block.adaLN_modulation[-1].weight, 0
            )
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.nta_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.nta_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        time_aligned_context: torch.Tensor,
        context: torch.Tensor,
        x_mask=None,
        context_mask=None
    ) -> torch.Tensor:

        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0]
                                        ).to(x.device, dtype=x.dtype)

        latent = self.input_proj(x.transpose(1, 2))  # (B, N, D)
        context = self.nta_proj(context)
        time_aligned_context = self.ta_proj(time_aligned_context)

        global_c = self.nta_global_proj(context.mean(dim=1))

        global_c = self.t_embed(timesteps).unsqueeze(1) + global_c.unsqueeze(
            1
        )  # (B, D)
        extended_c = global_c + time_aligned_context

        self.update_seq_lengths(latent.shape[1])

        for block in self.joint_blocks:
            latent, context = block(
                latent,
                context,
                global_c,
                extended_c,
                self.latent_rot,
                latent_mask=x_mask,
                nta_content_mask=context_mask,
            )

        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot, mask=x_mask)

        flow = self.final_layer(latent, extended_c)
        return flow.transpose(1, 2)

    @property
    def device(self) -> torch.device:
        return self.nta_global_proj.weight.device
