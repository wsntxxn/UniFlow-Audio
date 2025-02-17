from typing import Any, Literal, Callable
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torchaudio
from alias_free_torch import Activation1d

from models.common import LoadPretrainedBase
from models.autoencoder.autoencoder_base import AutoEncoderBase
from utils.torch_utilities import remove_key_prefix_factory, create_mask_from_length


# jit script make it 1.4x faster and save GPU memory
@torch.jit.script
def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta+0.000000001)) * pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=True
    ):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        # self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def get_activation(
    activation: Literal["elu", "snake", "none"],
    antialias=False,
    channels=None
) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation,
        use_snake=False,
        antialias_activation=False
    ):
        super().__init__()

        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels
            ),
            WNConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        res = x

        #x = checkpoint(self.layers, x)
        x = self.layers(x)

        return x + res


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False
    ):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=1,
                use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=3,
                use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=9,
                use_snake=use_snake
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2)
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False
    ):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding='same'
                )
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2)
            )

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels
            ),
            upsample_layer,
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=1,
                use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=3,
                use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=9,
                use_snake=use_snake
            ),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False
    ):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3
            )
        ]

        for i in range(self.depth - 1):
            layers += [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1
            )
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True
    ):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3
            ),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError


@torch.jit.script
def vae_sample(mean, scale) -> dict[str, torch.Tensor]:
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean*mean + var - logvar - 1).sum(1).mean()
    return {"latents": latents, "kl": kl}


class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self,
               x,
               return_info=False,
               **kwargs) -> dict[str, torch.Tensor] | torch.Tensor:
        mean, scale = x.chunk(2, dim=1)
        sampled = vae_sample(mean, scale)

        if return_info:
            return sampled["latents"], {"kl": sampled["kl"]}
        else:
            return sampled["latents"]

    def decode(self, x):
        return x


def compute_mean_kernel(x, y):
    kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
    return torch.exp(-kernel_input).mean()


class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def tokenize(self, x):
        raise NotImplementedError

    def decode_tokens(self, tokens):
        raise NotImplementedError


class StableVAE(LoadPretrainedBase, AutoEncoderBase):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels=None,
        out_channels=None,
        soft_clip=False,
        pretrained_ckpt: str | Path = None
    ):
        LoadPretrainedBase.__init__(self)
        AutoEncoderBase.__init__(
            self,
            downsampling_ratio=downsampling_ratio,
            sample_rate=sample_rate,
            latent_shape=(latent_dim, None)
        )

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels
        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.pretransform = pretransform
        self.soft_clip = soft_clip
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

        self.remove_autoencoder_prefix_fn: Callable = remove_key_prefix_factory(
            "autoencoder."
        )
        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt)

    def process_state_dict(self, model_dict, state_dict):
        state_dict = state_dict["state_dict"]
        state_dict = self.remove_autoencoder_prefix_fn(model_dict, state_dict)
        return state_dict

    def encode(
        self, waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(waveform)
        z = self.bottleneck.encode(z)
        z_length = waveform_lengths // self.downsampling_ratio
        z_mask = create_mask_from_length(z_length)
        return z, z_mask

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        waveform = self.decoder(latents)
        return waveform


if __name__ == '__main__':
    import hydra
    from utils.config import generate_config_from_command_line_overrides
    model_config = generate_config_from_command_line_overrides(
        "configs/model/autoencoder/stable_vae.yaml"
    )
    autoencoder: StableVAE = hydra.utils.instantiate(model_config)
    autoencoder.eval()

    waveform, sr = torchaudio.load(
        "/hpc_stor03/sjtu_home/xuenan.xu/workspace/singing_voice_synthesis/diffsinger/data/raw/opencpop/segments/wavs/2007000230.wav"
    )
    waveform = torchaudio.functional.resample(
        waveform, sr, model_config["sample_rate"]
    )
    print("waveform: ", waveform.shape)
    with torch.no_grad():
        latent, latent_length = autoencoder.encode(
            waveform, torch.as_tensor([waveform.shape[-1]])
        )
        print("latent: ", latent.shape)
        reconstructed = autoencoder.decode(latent)
        print("reconstructed: ", reconstructed.shape)
    import soundfile as sf
    sf.write(
        "./reconstructed.wav",
        reconstructed[0, 0].numpy(),
        samplerate=model_config["sample_rate"]
    )
