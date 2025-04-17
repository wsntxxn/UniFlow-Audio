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
        # super(LayerNorm, self)显式指定父类
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
        in_channels,
        filter_channels,
        n_layers=2,
        kernel_size=3,
        p_dropout=0.1,
        padding="SAME"
    ):
        super(DurationPredictor, self).__init__()
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = in_channels if idx == 0 else filter_channels
            self.conv += [
                nn.Sequential(
                    nn.ConstantPad1d(((kernel_size-1) // 2, (kernel_size-1) //
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


class ContentAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int,
        num_layers: int,
        num_heads: int,
        duration_predictor: DurationPredictor,
        dropout=0.1
    ):
        super().__init__()
        self.d_out = d_out
        self.d_model=d_model
        # [CLS]（classification）token 是 BERT 预训练时在每个输入序列的起始位置添加的一个特殊标记，主要用于 分类任务和全局特征提取
        # nn.Parameter 用于定义 可训练的模型参数，它会自动被 model.parameters() 识别，并在 optimizer 训练时更新。
        #d_model=256 初始化为256维的0向量
        self.cls_embed = nn.Parameter(torch.zeros((d_model)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        #定义transformer编码器，默认输入为 (seq_len, batch_size, d_model)，使用batch_first=True时，输入的形状为 (batch_size, seq_len, d_model)。
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers)
        self.duration_predictor = duration_predictor
        #定义卷积层，从d_model维到d_out维，1表示卷积核大小，因此实际上就是线性变换
        self.content_proj = torch.nn.Conv1d(d_model, d_out, 1)

    def forward(self, x, x_mask,task_emb=None):


        # ---------------------------------------------------------------------------- #
        #                                  拼接cls_emb ，task_emb                            #
        # ---------------------------------------------------------------------------- #
        #x: 形状 (batch, seq_len, d_model)，表示输入的序列特征。d_model 是特征维度,为256
	    #x_mask: 形状 (batch, seq_len)，用于指示哪些位置是有效的（1 表示有效，0 表示填充 PAD）。
        batch_size = x.size(0)
        # self.cls_embed 是一个 (d_model,) 的可训练向量（一个 CLS token）。
        #reshape(1, -1) 将其变为 (1, d_model)，适用于 batch 处理。
        #expand(batch_size, -1) 复制该 CLS token，使其形状变为 (batch_size, d_model)，保证每个样本都有相同的 CLS 。
        #cls_embed：[bs,1,d_model]
        cls_embed = self.cls_embed.reshape(1, -1).expand(batch_size, -1)
        # .unsqueeze(1) → 在 dim=1 处增加一个维度，使其形状从 (batch_size, d_model) 变为 (batch_size, 1, d_model)，
        cls_embed = cls_embed.to(x.device).unsqueeze(1)
        # cls_embed 作为第一个 token，拼接到 x 之前，形成新的 x，形状变为 (batch, seq_len + 1, d_model)。
        x = torch.cat([cls_embed, x], dim=1)
        # 创建 cls_mask（全为 1），形状 (batch, 1)，表示 [CLS] token 是有效的。
        cls_mask = torch.ones(batch_size, 1).to(x_mask.device)
        # 拼接 x_mask，使其形状变为 (batch, seq_len + 1)。
        x_mask = torch.cat([cls_mask, x_mask], dim=1)

        

        # ---------------------------------------------------------------------------- #
        #                                  拼接task_meb                                  #
        # ---------------------------------------------------------------------------- #
        if task_emb!=None:
            assert task_emb.shape[-1]%self.d_model==0
            task_meb=task_emb.reshape(batch_size,-1,self.d_model)
            task_emb_seq_len=task_emb.size(1)
            x=torch.cat([task_emb,x],dim=1)
            task_emb_mask=torch.ones(batch_size,task_emb_seq_len).to(x_mask.device)
            x_mask=torch.cat([task_emb_mask,x_mask],dim=1)
        else:
            task_emb_seq_len=0
        
        
        # ---------------------------------------------------------------------------- #
        #                            content过transformer编码器，然后抛弃task_emb                       #
        # ---------------------------------------------------------------------------- #
        x = self.encoder_layers(x, src_key_padding_mask=~x_mask.bool())
        x=x[:,task_emb_seq_len:]
        x_mask = x_mask[:, task_emb_seq_len:]
        
        
        # ---------------------------------------------------------------------------- #
        #                             过 duration predictor，直接输出为# [B, T+1, 1]                          #
        # ---------------------------------------------------------------------------- #
        x_detached = torch.detach(x)
        duration = self.duration_predictor(x_detached, x_mask).squeeze(-1)
        
        
        # x.transpose(1, 2)	交换 1维和2维(batch_size, seq_len, d_model) → (batch_size, d_model, seq_len)，方便进行线性变换
        content = self.content_proj(x.transpose(1, 2)).transpose(1, 2)
        #返回content,content_mask,global_duration_pred,local_duration_pred,形状分别为(batch, seq_len, d_out), (batch, seq_len), (batch,), (batch, seq_len)
        return content[:, 1:], x_mask[:, 1:], duration[:, 0], duration[:, 1:]
