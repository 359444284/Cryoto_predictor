import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.init import trunc_normal_

from .BasicModule import BasicModule
from torch.nn.utils import weight_norm
# 0.7351
from .layers.Embed import DataEmbedding, DataEmbedding_wo_temp
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, Destationary_Encoder, \
    Destationaty_EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, Flow_Attention_Causal, Destationary_Attention, \
    Destationary_AttentionLayer


class Learned_Aggregation_Layer(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


# class CausalConv1d(torch.nn.Conv1d):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  dilation=1,
#                  groups=1,
#                  bias=True):
#         super(CausalConv1d, self).__init__(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=0,
#             dilation=dilation,
#             groups=groups,
#             bias=bias)
#
#         self.__padding = (kernel_size - 1) * dilation
#
#     def forward(self, input):
#         return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

class Deeplob_CNN(nn.Module):
    def __init__(self):
        super(Deeplob_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )


    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.squeeze()
        return x.permute(0, 2, 1)


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         if d_model % 2 != 0:
#             pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]
#         else:
#             pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

# def forward(self, x):
#     """
#     Args:
#         x: Tensor, shape [seq_len, batch_size, embedding_dim]
#     """
#     x = x + self.pe[:x.size(0)]
#     return x

# https://github.com/ctxj/Time-Series-Transformer-Pytorch/blob/main/transformer_model.ipynb
class TransformerEn(BasicModule):
    def __init__(self):
        super().__init__()
        self.model_name = 'transformer_en'
        self.src_mask = None
        if self.config.use_time_feature:
            print('use_time_feature')
            self.enc_embedding = DataEmbedding(self.config.feature_dim, self.config.tran_emb_dim, 'timeF', 's',
                                            0.1, lockback=self.config.lockback_window)
        else:
            self.enc_embedding = DataEmbedding_wo_temp(self.config.feature_dim, self.config.tran_emb_dim, 'timeF', 's',
                                            0.1)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, self.config.factor, attention_dropout=self.config.tran_drop,
                                      output_attention=True),
                        # Flow_Attention_Causal(),
                        self.config.tran_emb_dim,
                        self.config.tran_num_head,
                    ),
                    self.config.tran_emb_dim,
                    self.config.tran_fc_dim,
                    dropout=self.config.tran_drop,
                    activation='gelu',
                    use_se=self.config.use_channel_att
                ) for l in range(self.config.tran_layer)
            ],
            norm_layer=torch.nn.LayerNorm(self.config.tran_emb_dim)
        )

        self.d_model = self.config.tran_emb_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.tran_emb_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.cls_norm = torch.nn.LayerNorm(self.config.tran_emb_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(self.config.tran_emb_dim*100, self.config.tran_emb_dim),
            nn.GELU()
        )
        self.fc2 = nn.Linear(self.config.tran_emb_dim, self.config.forecast_horizon // self.config.forecast_stride)
        self.aggre = Learned_Aggregation_Layer(dim=self.config.tran_emb_dim, num_heads=self.config.tran_num_head)
        self.drop = nn.Dropout(0.3)

    def forward(self, src, scr_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feature_dim]

        Returns:
            output Tensor of shape [batch_size , n_class]
        """

        B, L, C = src.shape

        if self.src_mask is None or self.src_mask.size(0) != L:
            mask = self._generate_square_subsequent_mask(L).to(self.device)
            self.src_mask = mask

        enc_out = self.enc_embedding(src, scr_mask)

        enc_out, attns = self.encoder(enc_out, attn_mask=self.src_mask)

        output = enc_out

        #  CLS header
        # cls_token = self.cls_token.expand(B, 1, -1)
        # output = torch.cat([cls_token, output], dim=1)
        # output = self.aggre(output)

        output = torch.flatten(output, start_dim=1)
        output = self.drop(self.fc1(output))
        output = self.fc2(output)
        return output, attns

    def _generate_square_subsequent_mask(self, sz):
        # mask_shape = [B, 1, sz, sz]
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
