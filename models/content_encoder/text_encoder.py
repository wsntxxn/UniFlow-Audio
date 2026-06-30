from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    DEVICE_TYPE = "npu"
except ModuleNotFoundError:
    DEVICE_TYPE = "cuda"


class TransformersTextEncoderBase(nn.Module):
    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.hidden_size, embed_dim)

    def forward(
        self,
        text: list[str],
    ):
        output, mask = self.encode(text)
        output = self.projection(output)
        return {"output": output, "mask": mask}

    def encode(self, text: list[str]):
        device = self.model.device
        batch = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        output: BaseModelOutput = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = output.last_hidden_state
        mask = (attention_mask == 1).to(device)
        return output, mask

    def projection(self, x):
        return self.proj(x)


class T5TextEncoder(TransformersTextEncoderBase):
    def __init__(
        self, embed_dim: int, model_name: str = "google/flan-t5-large"
    ):
        nn.Module.__init__(self)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.proj = nn.Linear(self.model.config.hidden_size, embed_dim)

    def encode(
        self,
        text: list[str],
    ):
        with torch.no_grad(), torch.amp.autocast(
            device_type=DEVICE_TYPE, enabled=False
        ):
            return super().encode(text)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0
):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor**(dim / (dim - 2))
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=padding,
            groups=dim,
            dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNeXtTextEncoder(nn.Module):
    def __init__(
        self,
        text_num_embeds,
        text_dim,
        d_out,
        mask_padding=True,
        average_upsampling=False,
        conv_layers=0,
        conv_mult=2
    ):
        super().__init__()
        self.text_embed = nn.Embedding(
            text_num_embeds + 1, text_dim
        )  # use 0 as filler token
        self.out_proj = nn.Linear(text_dim, d_out)
        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 is ~87.38s of 24khz audio; 4096 is ~43.69s of 24khz audio
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False
            )
            self.text_blocks = nn.Sequential(
                *[
                    ConvNeXtV2Block(text_dim, text_dim * conv_mult)
                    for _ in range(conv_layers)
                ]
            )
        else:
            self.extra_modeling = False

    def average_upsample_text_by_mask(self, text, text_mask):
        batch, text_len, text_dim = text.shape

        audio_len = text_len  # cuz text already padded to same length as audio sequence
        text_lens = text_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            text_len = text_lens[i].item()

            if text_len == 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [text_len, text_dim]

            base_repeat = audio_len // text_len
            remainder = audio_len % text_len

            indices = []
            for j in range(text_len):
                repeat_count = base_repeat + (
                    1 if j >= text_len - remainder else 0
                )
                indices.extend([j] * repeat_count)

            indices = torch.tensor(
                indices[:audio_len], device=text.device, dtype=torch.long
            )
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(self, phoneme, phoneme_lengths):
        device = next(self.parameters()).device
        phoneme = phoneme.to(device)
        phoneme_lengths = phoneme_lengths.to(device)
        seq_len = int(phoneme_lengths.max().item())
        max_length = phoneme.shape[1]
        assert seq_len == max_length
        if self.mask_padding:
            phoneme_mask = phoneme == 0

        phoneme = self.text_embed(phoneme)  # b n -> b n d
        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            phoneme = phoneme + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                phoneme = phoneme.masked_fill(
                    phoneme_mask.unsqueeze(-1).expand(
                        -1, -1, phoneme.size(-1)
                    ), 0.0
                )
                for block in self.text_blocks:
                    phoneme = block(phoneme)
                    phoneme = phoneme.masked_fill(
                        phoneme_mask.unsqueeze(-1).expand(
                            -1, -1, phoneme.size(-1)
                        ), 0.0
                    )
            else:
                phoneme = self.text_blocks(phoneme)

        if self.average_upsampling:
            phoneme = self.average_upsample_text_by_mask(
                phoneme, ~phoneme_mask
            )
        phoneme = self.out_proj(phoneme)
        return {
            "output": phoneme,
            "mask": ~phoneme_mask,
        }


if __name__ == "__main__":
    text_encoder = T5TextEncoder(embed_dim=512)
    text = ["a man is speaking", "a woman is singing while a dog is barking"]

    output = text_encoder(text)
