from typing import Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_utilities import create_mask_from_length
from utils.diffsinger_utilities import denorm_f0, f0_to_coarse


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) *
            mask).long() + padding_idx


class PositionalEncoding(nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """
    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1),
                                    dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """
    def __init__(self, d_model, padding_idx, init_size=2048):
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            d_model,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, d_model, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings,
                           dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if d_model % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        x,
        lengths,
        incremental_state=None,
        timestep=None,
        positions=None,
        **kwargs
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = x.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.d_model,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = create_mask_from_length(
            lengths, max_length=x.shape[1]
        ) * (torch.arange(x.shape[1]) + 1).unsqueeze(0).expand(x.shape[0], -1)
        positions = positions.to(self.weights.device)
        pos_emb = self.weights.index_select(0, positions.view(-1)).view(
            bsz, seq_len, -1
        ).detach()
        return x + pos_emb

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """
    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x, lengths):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, :x.size(1)]
        return self.dropout(x) + self.dropout(pos_emb)


class TransformerFFNLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        filter_size,
        padding="SAME",
        kernel_size=1,
        dropout=0.,
        act='gelu'
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(
                hidden_size,
                filter_size,
                kernel_size,
                padding=kernel_size // 2
            )
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(hidden_size, filter_size, kernel_size)
            )
        self.ffn_2 = nn.Linear(filter_size, hidden_size)

    def forward(
        self,
        x,
    ):
        # x: T x B x C
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size**-0.5

        if self.act == 'gelu':
            x = F.gelu(x)
        if self.act == 'relu':
            x = F.relu(x)
        if self.act == 'swish':
            x = F.silu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class EncoderSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        c,
        num_heads,
        dropout,
        attention_dropout=0.1,
        relu_dropout=0.1,
        kernel_size=9,
        padding='SAME',
        act='gelu'
    ):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            self.layer_norm1 = nn.LayerNorm(c)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=self.c,
                num_heads=num_heads,
                dropout=attention_dropout,
                bias=False,
            )
        self.layer_norm2 = nn.LayerNorm(c)
        self.ffn = TransformerFFNLayer(
            c,
            4 * c,
            kernel_size=kernel_size,
            dropout=relu_dropout,
            padding=padding,
            act=act
        )

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x)
            x, _, = self.self_attn(
                query=x, key=x, value=x, key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[...,
                                                                       None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        kernel_size,
        num_heads=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncoderSelfAttentionLayer(
            hidden_size,
            num_heads,
            dropout=dropout,
            attention_dropout=0.0,
            relu_dropout=dropout,
            kernel_size=kernel_size,
            padding="SAME",
            act="gelu"
        )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class FFTBlocks(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        ffn_kernel_size=9,
        dropout=0.1,
        num_heads=2,
        use_last_norm=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_last_norm = use_last_norm

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(
                self.hidden_size,
                self.dropout,
                kernel_size=ffn_kernel_size,
                num_heads=num_heads
            ) for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, padding_mask=None, attn_mask=None):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        if padding_mask is None:
            padding_mask = torch.zeros(x.size(0), x.size(1)).to(x.device)
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float(
        )[:, :, None]  # [T, B, 1]
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=padding_mask, attn_mask=attn_mask
            ) * nonpadding_mask_TB
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB

        x = x.transpose(0, 1)  # [B, T, C]
        return x


class FastSpeech2EncoderBase(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_kernel_size: int,
        d_out: int,
        dropout: float = 0.1,
        rel_pos: bool = True
    ):
        super().__init__()
        self.rel_pos = rel_pos

        if self.rel_pos:
            self.pos_encoding = RelPositionalEncoding(
                d_model, dropout_rate=0.0
            )
        else:
            self.pos_encoding = SinusoidalPositionalEmbedding(
                d_model, padding_idx=0
            )
        self.dropout = dropout
        self.embed_scale = math.sqrt(d_model)

        self.layers = FFTBlocks(
            hidden_size=d_model,
            num_layers=num_layers,
            ffn_kernel_size=ffn_kernel_size,
            dropout=dropout,
            num_heads=num_heads,
            use_last_norm=True
        )

        self.out_proj = nn.Linear(d_model, d_out)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=m.embedding_dim**-0.5)


class FastSpeech2MIDIEncoder(FastSpeech2EncoderBase):
    def __init__(
        self,
        phone_vocab_size: int,
        midi_vocab_size: int,
        slur_vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_kernel_size: int,
        d_out: int,
        dropout: float = 0.1,
        rel_pos: bool = True
    ):
        super().__init__(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_kernel_size=ffn_kernel_size,
            d_out=d_out,
            dropout=dropout,
            rel_pos=rel_pos
        )
        self.phone_embed = nn.Embedding(phone_vocab_size, d_model)
        self.midi_embed = nn.Embedding(midi_vocab_size, d_model, padding_idx=0)
        self.midi_dur_embed = nn.Linear(1, d_model)
        self.is_slur_embed = nn.Embedding(slur_vocab_size, d_model)

    def forward(
        self, phoneme: torch.Tensor, midi: torch.Tensor,
        midi_duration: torch.Tensor, is_slur: torch.Tensor,
        lengths: Sequence[int]
    ):
        x = self.embed_scale * self.phone_embed(phoneme)
        midi_embedding = self.midi_embed(midi)
        midi_dur_embedding = self.midi_dur_embed(midi_duration[:, :, None])
        slur_embedding = self.is_slur_embed(is_slur)

        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        x = self.pos_encoding(x, lengths)
        x = F.dropout(x, p=self.dropout, training=self.training)

        padding_mask = ~create_mask_from_length(lengths).to(phoneme.device)
        x = self.layers(x, padding_mask=padding_mask)

        x = self.out_proj(x)

        return {"output": x, "mask": ~padding_mask}


class FastSpeech2PitchEncoder(FastSpeech2EncoderBase):
    def __init__(
        self,
        phone_vocab_size,
        d_model,
        num_layers,
        num_heads,
        ffn_kernel_size,
        d_out,
        dropout=0.1,
        rel_pos=False
    ):
        super().__init__(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_kernel_size=ffn_kernel_size,
            d_out=d_out,
            dropout=dropout,
            rel_pos=rel_pos
        )
        self.phone_embed = nn.Embedding(phone_vocab_size, d_model)
        self.pitch_embed = nn.Embedding(300, d_model)

    def forward(self, phoneme: torch.Tensor, lengths: Sequence[int]):
        x = self.embed_scale * self.phone_embed(phoneme)
        x = self.pos_encoding(x, lengths)
        x = F.dropout(x, p=self.dropout, training=self.training)

        padding_mask = ~create_mask_from_length(lengths).to(phoneme.device)
        x = self.layers(x, padding_mask=padding_mask)

        x = self.out_proj(x)

        return {"output": x, "mask": ~padding_mask}

    def encode_pitch(self, f0, uv):

        f0_denorm = denorm_f0(f0, uv)
        pitch = f0_to_coarse(f0_denorm)
        pitch_embed = self.pitch_embed(pitch)
        return {"output": pitch_embed}
