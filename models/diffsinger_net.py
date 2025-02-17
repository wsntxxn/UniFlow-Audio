import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(
            residual_channels, residual_channels
        )
        self.conditioner_projection = nn.Conv1d(
            encoder_hidden, 2 * residual_channels, 1
        )
        self.output_projection = nn.Conv1d(
            residual_channels, 2 * residual_channels, 1
        )

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step
                                                  ).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x+residual) / math.sqrt(2.0), skip


class DiffSingerNet(nn.Module):
    def __init__(
        self,
        in_dims=128,
        residual_channels=256,
        encoder_hidden=256,
        dilation_cycle_length=4,
        residual_layers=20,
    ):
        super().__init__()

        # self.pe_scale = pe_scale

        self.input_projection = nn.Conv1d(in_dims, residual_channels, 1)
        self.time_pos_emb = SinusoidalPosEmb(residual_channels)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden, residual_channels,
                2**(i % dilation_cycle_length)
            ) for i in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(
            residual_channels, residual_channels, 1
        )
        self.output_projection = nn.Conv1d(residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, timesteps, context, x_mask=None, context_mask=None):
        # make it compatible with int time step during inference
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0]
                                        ).to(x.device, dtype=torch.long)

        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)

        t = self.time_pos_emb(timesteps)
        t = self.mlp(t)

        cond = context

        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, t)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip),
                      dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, M, T]
        return x * x_mask.unsqueeze(1)
