import torch
import torch.nn as nn


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
                    nn.ConstantPad1d(
                        ((kernel_size - 1) // 2,
                         (kernel_size - 1) // 2) if padding == 'SAME' else
                        (kernel_size - 1, 0), 0
                    ),
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
        d_model: int,
        d_out: int,
        prefix_dim: int,
        num_layers: int,
        num_heads: int,
        duration_predictor: DurationPredictor,
        dropout: float = 0.1,
        norm_first: bool = False,
        activation: str = "gelu",
        duration_grad_scale: float = 0.1,
    ):
        super().__init__(d_out)
        self.duration_grad_scale = duration_grad_scale
        self.prefix_mlp = nn.Sequential(
            nn.Linear(prefix_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
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
        self.layers = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            enable_nested_tensor=enable_nested_tensor
        )
        self.duration_predictor = duration_predictor
        self.content_proj = nn.Conv1d(d_model, d_out, 1)

    def forward(self, content, content_mask, instruction, instruction_mask):
        batch_size = content.size(0)
        cls_embed = self.cls_embed.reshape(1, -1).expand(batch_size, -1)
        cls_embed = cls_embed.to(content.device).unsqueeze(1)
        x = torch.cat([cls_embed, content], dim=1)
        cls_mask = torch.ones(batch_size, 1,
                              dtype=bool).to(content_mask.device)
        x_mask = torch.cat([cls_mask, content_mask], dim=1)

        prefix = self.prefix_mlp(instruction)
        seq = torch.cat([prefix, x], dim=1)
        seq_mask = torch.cat([instruction_mask, x_mask], dim=1)
        x = self.layers(seq, src_key_padding_mask=~seq_mask.bool())
        x = x[:, prefix.size(1):, :]

        x_grad_rescaled = x * self.duration_grad_scale + x.detach(
        ) * (1 - self.duration_grad_scale)
        duration = self.duration_predictor(x_grad_rescaled, x_mask).squeeze(-1)
        content = self.content_proj(x.transpose(1, 2)).transpose(1, 2)
        return content[:, 1:], x_mask[:, 1:], duration[:, 0], duration[:, 1:]
