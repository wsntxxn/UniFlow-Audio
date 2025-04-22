from typing import Sequence
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

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


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def LayerNorm(
    normalized_shape, eps=1e-5, elementwise_affine=True, export=False
):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class BatchNorm1dTBC(nn.Module):
    def __init__(self, c):
        super(BatchNorm1dTBC, self).__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        """

        :param x: [T, B, C]
        :return: [T, B, C]
        """
        x = x.permute(1, 2, 0)  # [B, C, T]
        x = self.bn(x)  # [B, C, T]
        x = x.permute(2, 0, 1)  # [T, B, C]
        return x


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
        emb = math.log(10000) / (half_dim - 1)
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


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(
                torch.Tensor(3 * embed_dim, embed_dim)
            )
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.last_attn_probs = None

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
        enc_dec_attn_constraint_mask=None,
        reset_attn_weight=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.enable_torch_version and incremental_state is None and not static_kv and reset_attn_weight is None:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.bias_k,
                    self.bias_v, self.add_zero_attn, self.dropout,
                    self.out_proj.weight, self.out_proj.bias, self.training,
                    key_padding_mask, need_weights, attn_mask
                )
            else:
                return F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    self.in_proj_bias,
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training,
                    key_padding_mask,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight,
                    k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight
                )

        if incremental_state is not None:
            print('Not implemented error.')
            exit()
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask,
                     attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(
                            key_padding_mask.size(0), 1
                        )
                    ],
                    dim=1
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)

        if saved_state is not None:
            print('Not implemented error.')
            exit()

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size(
            []
        ):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat(
                [k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1
            )
            v = torch.cat(
                [v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1
            )
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask,
                     attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0),
                                    1).type_as(key_padding_mask)
                    ],
                    dim=1
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz
        )

        assert list(attn_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len
        ]

        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask[:, None].repeat(
                    [1, self.num_heads, 1, 1]
                ).reshape(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attn_mask

        if enc_dec_attn_constraint_mask is not None:  # bs x head x L_kv
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.masked_fill(
                enc_dec_attn_constraint_mask.unsqueeze(2).bool(),
                -1e9,
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -1e9,
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training
        )

        if reset_attn_weight is not None:
            if reset_attn_weight:
                self.last_attn_probs = attn_probs.detach()
            else:
                assert self.last_attn_probs is not None
                attn_probs = self.last_attn_probs
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [
            bsz * self.num_heads, tgt_len, self.head_dim
        ]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, (attn_weights, attn_logits)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(
                key, start=self.embed_dim, end=2 * self.embed_dim
            )
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights


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
        norm='ln',
        act='gelu',
        padding_set_zero=True
    ):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        self.padding_set_zero = padding_set_zero
        if num_heads > 0:
            if norm == 'ln':
                self.layer_norm1 = LayerNorm(c)
            elif norm == 'bn':
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = MultiheadAttention(
                self.c,
                num_heads=num_heads,
                self_attention=True,
                dropout=attention_dropout,
                bias=False,
            )
        if norm == 'ln':
            self.layer_norm2 = LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm2 = BatchNorm1dTBC(c)
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
            if self.padding_set_zero:
                x = x * (1 - encoder_padding_mask.float()).transpose(0,
                                                                     1)[...,
                                                                        None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        if self.padding_set_zero:
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[...,
                                                                       None]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        kernel_size,
        num_heads=2,
        norm='ln',
        padding_set_zero=True,
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
            norm=norm,
            act="gelu",
            padding_set_zero=padding_set_zero
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
        padding_set_zero=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_last_norm = use_last_norm
        self.padding_set_zero = padding_set_zero

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    self.hidden_size,
                    self.dropout,
                    kernel_size=ffn_kernel_size,
                    num_heads=num_heads,
                    padding_set_zero=padding_set_zero,
                ) for _ in range(self.num_layers)
            ]
        )
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
        x = x.transpose(0, 1)
        if self.padding_set_zero:
            x = x * nonpadding_mask_TB
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=padding_mask, attn_mask=attn_mask
            )
            if self.padding_set_zero:
                x = x * nonpadding_mask_TB
        if self.use_last_norm:
            x = self.layer_norm(x)
            if self.padding_set_zero:
                x = x * nonpadding_mask_TB

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
        rel_pos: bool = True,
        padding_set_zero: bool = True
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
            use_last_norm=True,
            padding_set_zero=padding_set_zero
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


@dataclass
class SpkConfig:
    encoding_format: str
    num_spk: int | None = None
    spk_embed_dim: int | None = None

    def __post_init__(self):
        allowed_formats = {"id", "embedding"}
        assert self.encoding_format in allowed_formats, f"mode must be one of {allowed_formats}, got '{self.encoding_format}'"
        if self.encoding_format == "id":
            assert self.num_spk is not None
        if self.encoding_format == "embedding":
            assert self.spk_embed_dim is not None


class FastSpeech2PhonemeEncoder(FastSpeech2EncoderBase):
    def __init__(
        self,
        phone_vocab_size,
        d_model,
        num_layers,
        num_heads,
        ffn_kernel_size,
        d_out,
        dropout=0.1,
        rel_pos=False,
        spk_config: SpkConfig | None = None,
        padding_set_zero: bool = True
    ):
        super().__init__(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_kernel_size=ffn_kernel_size,
            d_out=d_out,
            dropout=dropout,
            rel_pos=rel_pos,
            padding_set_zero=padding_set_zero
        )
        self.phone_embed = Embedding(phone_vocab_size, d_model)
        self.spk_config = spk_config
        if spk_config is not None:
            if spk_config.encoding_format == "id":
                self.spk_embed_proj = Embedding(
                    spk_config.num_spk + 1, d_model
                )
            elif spk_config.encoding_format == "embedding":
                self.spk_embed_proj = Linear(spk_config.spk_embed_dim, d_model)

    def forward(
        self, phoneme: torch.Tensor, lengths: Sequence[int], spk: torch.Tensor
    ):
        x = self.embed_scale * self.phone_embed(phoneme)
        x = self.pos_encoding(x, lengths)
        x = F.dropout(x, p=self.dropout, training=self.training)

        padding_mask = ~create_mask_from_length(lengths).to(phoneme.device)
        x = self.layers(x, padding_mask=padding_mask)

        if self.spk_config is not None:
            spk_embed = self.spk_embed_proj(spk).unsqueeze(1)
            x = x + spk_embed

        x = self.out_proj(x)

        return {"output": x, "mask": ~padding_mask}


class FastSpeech2MIDIEncoder(FastSpeech2PhonemeEncoder):
    def __init__(
        self,
        phone_vocab_size: int,
        midi_vocab_size: int,
        slur_vocab_size: int,
        spk_config: SpkConfig | None,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_kernel_size: int,
        d_out: int,
        dropout: float = 0.1,
        rel_pos: bool = True,
        padding_set_zero: bool = True
    ):
        super().__init__(
            phone_vocab_size=phone_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_kernel_size=ffn_kernel_size,
            d_out=d_out,
            dropout=dropout,
            rel_pos=rel_pos,
            spk_config=spk_config,
            padding_set_zero=padding_set_zero
        )
        self.midi_embed = Embedding(midi_vocab_size, d_model, padding_idx=0)
        self.midi_dur_embed = Linear(1, d_model)
        self.is_slur_embed = Embedding(slur_vocab_size, d_model)

    def forward(
        self,
        phoneme: torch.Tensor,
        midi: torch.Tensor,
        midi_duration: torch.Tensor,
        is_slur: torch.Tensor,
        lengths: Sequence[int],
        spk: torch.Tensor | None = None,
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

        if self.spk_config is not None:
            spk_embed = self.spk_embed_proj(spk).unsqueeze(1)
            x = x + spk_embed

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
        rel_pos=False,
        padding_set_zero=True
    ):
        super().__init__(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_kernel_size=ffn_kernel_size,
            d_out=d_out,
            dropout=dropout,
            rel_pos=rel_pos,
            padding_set_zero=padding_set_zero
        )
        self.phone_embed = Embedding(phone_vocab_size, d_model)
        self.pitch_embed = Embedding(300, d_model)

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
