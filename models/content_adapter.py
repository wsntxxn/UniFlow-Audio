import math
import torch
import torch.nn as nn

from utils.torch_utilities import concat_non_padding, restore_from_concat


######################
# fastspeech modules
######################
class LayerNorm(nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """
    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm,
                     self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        padding: str = "SAME"
    ):
        super(DurationPredictor, self).__init__()
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = in_channels if idx == 0 else filter_channels
            self.conv += [
                nn.Sequential(
                    nn.ConstantPad1d(((kernel_size - 1) // 2,
                                      (kernel_size - 1) //
                                      2) if padding == 'SAME' else
                                     (kernel_size - 1, 0), 0),
                    nn.Conv1d(
                        in_chans,
                        filter_channels,
                        kernel_size,
                        stride=1,
                        padding=0
                    ), nn.ReLU(), LayerNorm(filter_channels, dim=1),
                    nn.Dropout(p_dropout)
                )
            ]
        self.linear = nn.Linear(filter_channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        # x: [B, T, E]
        x = x.transpose(1, -1)
        x_mask = x_mask.unsqueeze(1).to(x.device)
        for f in self.conv:
            x = f(x)
            x = x * x_mask.float()

        x = self.linear(x.transpose(1, -1)
                       ) * x_mask.transpose(1, -1).float()  # [B, T, 1]
        return x


######################
# adapter modules
######################


class ContentAdapterBase(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.d_out = d_out


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class ContentAdapter(ContentAdapterBase):
    def __init__(
        self,
        d_model: int,
        d_out: int,
        num_layers: int,
        num_heads: int,
        duration_predictor: DurationPredictor,
        dropout: float = 0.1,
        norm_first: bool = False,
        activation: str = "gelu",
        duration_grad_scale: float = 0.0,
    ):
        super().__init__(d_out)
        self.duration_grad_scale = duration_grad_scale
        self.cls_embed = nn.Parameter(torch.randn(d_model))
        if hasattr(torch, "npu") and torch.npu.is_available():
            enable_nested_tensor = False
        else:
            enable_nested_tensor = True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )
        self.encoder_layers = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=enable_nested_tensor
        )
        self.duration_predictor = duration_predictor
        self.content_proj = nn.Conv1d(d_model, d_out, 1)

    def forward(self, x, x_mask):
        batch_size = x.size(0)
        cls_embed = self.cls_embed.reshape(1, -1).expand(batch_size, -1)
        cls_embed = cls_embed.to(x.device).unsqueeze(1)
        x = torch.cat([cls_embed, x], dim=1)

        cls_mask = torch.ones(batch_size, 1).to(x_mask.device)
        x_mask = torch.cat([cls_mask, x_mask], dim=1)
        x = self.encoder_layers(x, src_key_padding_mask=~x_mask.bool())
        x_grad_rescaled = x * self.duration_grad_scale + x.detach(
        ) * (1 - self.duration_grad_scale)
        duration = self.duration_predictor(x_grad_rescaled, x_mask).squeeze(-1)
        content = self.content_proj(x.transpose(1, 2)).transpose(1, 2)
        return content[:, 1:], x_mask[:, 1:], duration[:, 0], duration[:, 1:]


class PrefixAdapter(ContentAdapterBase):
    def __init__(
        self,
        content_dim: int,
        d_model: int,
        d_out: int,
        prefix_dim: int,
        num_layers: int,
        num_heads: int,
        duration_predictor: DurationPredictor,
        dropout: float = 0.1,
        norm_first: bool = False,
        use_last_norm: bool = True,
        activation: str = "gelu",
        duration_grad_scale: float = 0.1,
    ):
        super().__init__(d_out)
        self.duration_grad_scale = duration_grad_scale
        self.prefix_mlp = nn.Sequential(
            nn.Linear(prefix_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.content_mlp = nn.Sequential(
            nn.Linear(content_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first
        )
        if hasattr(torch, "npu") and torch.npu.is_available():
            enable_nested_tensor = False
        else:
            enable_nested_tensor = True
        self.cls_embed = nn.Parameter(torch.randn(d_model))
        # self.pos_embed = SinusoidalPositionalEmbedding(d_model, dropout)
        self.layers = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            enable_nested_tensor=enable_nested_tensor
        )
        self.use_last_norm = use_last_norm
        if self.use_last_norm:
            self.last_norm = nn.LayerNorm(d_model)
        self.duration_predictor = duration_predictor
        self.content_proj = nn.Conv1d(d_model, d_out, 1)
        nn.init.normal_(self.cls_embed, 0., 0.02)
        nn.init.xavier_uniform_(self.content_proj.weight)
        nn.init.constant_(self.content_proj.bias, 0.)

    def forward(self, content, content_mask, instruction, instruction_mask):
        batch_size = content.size(0)
        cls_embed = self.cls_embed.reshape(1, -1).expand(batch_size, -1)
        cls_embed = cls_embed.to(content.device).unsqueeze(1)
        content = self.content_mlp(content)
        x = torch.cat([cls_embed, content], dim=1)
        cls_mask = torch.ones(batch_size, 1,
                              dtype=bool).to(content_mask.device)
        x_mask = torch.cat([cls_mask, content_mask], dim=1)

        prefix = self.prefix_mlp(instruction)
        seq, seq_mask, perm = concat_non_padding(
            prefix, instruction_mask, x, x_mask
        )
        # seq = self.pos_embed(seq)
        x = self.layers(seq, src_key_padding_mask=~seq_mask.bool())
        if self.use_last_norm:
            x = self.last_norm(x)
        _, x = restore_from_concat(x, instruction_mask, x_mask, perm)

        x_grad_rescaled = x * self.duration_grad_scale + x.detach(
        ) * (1 - self.duration_grad_scale)
        duration = self.duration_predictor(x_grad_rescaled, x_mask).squeeze(-1)
        content = self.content_proj(x.transpose(1, 2)).transpose(1, 2)
        return content[:, 1:], x_mask[:, 1:], duration[:, 0], duration[:, 1:]


class CrossAttentionAdapter(ContentAdapterBase):
    def __init__(
        self,
        d_out: int,
        content_dim: int,
        prefix_dim: int,
        num_heads: int,
        duration_predictor: DurationPredictor,
        dropout: float = 0.1,
        duration_grad_scale: float = 0.1,
    ):
        super().__init__(d_out)
        self.attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=prefix_dim,
            vdim=prefix_dim,
            batch_first=True,
        )
        self.duration_grad_scale = duration_grad_scale
        self.duration_predictor = duration_predictor
        self.global_duration_mlp = nn.Sequential(
            nn.Linear(content_dim, content_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(content_dim, 1)
        )
        self.norm = nn.LayerNorm(content_dim)
        self.content_proj = nn.Conv1d(content_dim, d_out, 1)

    def forward(self, content, content_mask, prefix, prefix_mask):
        attn_output, attn_output_weights = self.attn(
            query=content,
            key=prefix,
            value=prefix,
            key_padding_mask=~prefix_mask.bool()
        )
        attn_output = attn_output * content_mask.unsqueeze(-1).float()
        x = self.norm(attn_output + content)
        x_grad_rescaled = x * self.duration_grad_scale + x.detach(
        ) * (1 - self.duration_grad_scale)
        x_aggregated = (x_grad_rescaled * content_mask.unsqueeze(-1).float()
                       ).sum(dim=1) / content_mask.sum(dim=1,
                                                       keepdim=True).float()
        global_duration = self.global_duration_mlp(x_aggregated).squeeze(-1)
        local_duration = self.duration_predictor(
            x_grad_rescaled, content_mask
        ).squeeze(-1)
        content = self.content_proj(x.transpose(1, 2)).transpose(1, 2)
        return content, content_mask, global_duration, local_duration


class ExperimentalCrossAttentionAdapter(ContentAdapterBase):
    def __init__(
        self,
        d_out: int,
        content_dim: int,
        prefix_dim: int,
        num_heads: int,
        duration_predictor: DurationPredictor,
        dropout: float = 0.1,
        duration_grad_scale: float = 0.1,
    ):
        super().__init__(d_out)
        self.content_mlp = nn.Sequential(
            nn.Linear(content_dim, content_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim, content_dim),
        )
        self.content_norm = nn.LayerNorm(content_dim)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(prefix_dim, prefix_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prefix_dim, prefix_dim),
        )
        self.prefix_norm = nn.LayerNorm(content_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=prefix_dim,
            vdim=prefix_dim,
            batch_first=True,
        )
        self.duration_grad_scale = duration_grad_scale
        self.duration_predictor = duration_predictor
        self.global_duration_mlp = nn.Sequential(
            nn.Linear(content_dim, content_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(content_dim, 1)
        )
        self.content_proj = nn.Sequential(
            nn.Linear(content_dim, d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )
        self.norm1 = nn.LayerNorm(content_dim)
        self.norm2 = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)

        self.apply(_init_weights)

    def forward(self, content, content_mask, prefix, prefix_mask):
        content = self.content_mlp(content)
        content = self.content_norm(content)
        prefix = self.prefix_mlp(prefix)
        prefix = self.prefix_norm(prefix)
        attn_output, attn_weights = self.attn(
            query=content,
            key=prefix,
            value=prefix,
            key_padding_mask=~prefix_mask.bool(),
        )
        attn_output = attn_output * content_mask.unsqueeze(-1).float()
        x = attn_output + content
        x = self.norm1(x)
        x_grad_rescaled = x * self.duration_grad_scale + x.detach(
        ) * (1 - self.duration_grad_scale)
        x_aggregated = (x_grad_rescaled * content_mask.unsqueeze(-1).float()
                       ).sum(dim=1) / content_mask.sum(dim=1,
                                                       keepdim=True).float()
        global_duration = self.global_duration_mlp(x_aggregated).squeeze(-1)
        local_duration = self.duration_predictor(
            x_grad_rescaled, content_mask
        ).squeeze(-1)
        content = self.content_proj(x)
        content = self.norm2(content)
        return content, content_mask, global_duration, local_duration
