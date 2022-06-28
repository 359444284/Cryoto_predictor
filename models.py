import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from einops.layers.torch import Rearrange

EPSILON = np.finfo(float).eps


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, w, c = x.size()
        y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y.expand_as(x)


class CNN(nn.Module):
    def __init__(self, input_dim, config):
        super(CNN, self).__init__()
        # (batch, 100, input_dim)
        self.config = config
        self.convs = nn.Sequential(
            nn.Conv1d(config.lockback_window, 16, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(3, 6),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(6, 3),
            nn.Conv1d(32, 32, 3, 1, 1)
        )
        self.fc = nn.Linear(96, config.forecast_horizon // config.forecast_stride)

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(torch.flatten(x, start_dim=1))

        return x


# adapted from <DeepLOB: Deep Convolutional Neural Networks
# for Limit Order Books>
# https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/blob/master/jupyter_pytorch/run_train_pytorch.ipynb
class deeplob(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.device = device
        self.config = config
        # convolution blocks
        # self.feature_select = SELayer(feature_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            #             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 4)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            # nn.AdaptiveAvgPool2d((None, 1))
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, int(self.config.forecast_horizon / self.config.forecast_stride))

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(self.device)
        c0 = torch.zeros(1, x.size(0), 64).to(self.device)
        # x = self.feature_select(x)
        x = torch.unsqueeze(x, 1)
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        # print(x.size())
        #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)

        return x


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.feature_dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, config.forecast_horizon)
        )

    def forward(self, x):
        # print(x.size())
        x = torch.squeeze(x, 1)
        x = self.fc(x)

        return x


class LSTM(nn.Module):
    def __init__(self, device, config, hidden_size=64, num_layer=2):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=config.feature_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layer,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.feature_select = SELayer(config.feature_dim)
        self.fc = nn.Linear(self.hidden_size, int(config.forecast_horizon / config.forecast_stride))

    def forward(self, x):
        # h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(self.device)
        x = self.feature_select(x)
        x, _ = self.lstm(x, None)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


# adapted from https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.0):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x

class MLPMixer(nn.Module):

    def __init__(self, config, dim=64, patch_size=4, depth=3, token_dim=128, channel_dim=256):
        super().__init__()
        assert config.lockback_window % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (config.lockback_window//patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(config.feature_dim, dim, (patch_size, 1), (patch_size, 1)),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, config.forecast_horizon//config.forecast_stride)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.unsqueeze(x, -1)

        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)


# https://github.com/ctxj/Time-Series-Transformer-Pytorch/blob/main/transformer_model.ipynb
class TransformerModel(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.config = config
        self.device = device
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(config.tran_emb_dim)
        encoder_layers = TransformerEncoderLayer(d_model=config.tran_emb_dim,
                                                 nhead=config.tran_num_head,
                                                 dim_feedforward=config.tran_fc_dim,
                                                 dropout=config.tran_drop)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.tran_layer)
        if config.tran_use_embed:
            self.encoder = nn.Linear(config.feature_dim, config.tran_emb_dim)
        self.d_model = config.tran_emb_dim
        self.fc1 = nn.Sequential(
            nn.Linear(config.tran_emb_dim * config.lockback_window, config.tran_emb_dim),
            nn.Tanh()
        )
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(config.tran_emb_dim, config.forecast_horizon // config.forecast_stride)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, feature_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, hidden]
        """
        src = src.permute(1, 0, 2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(self.device)
            self.src_mask = mask
        if self.config.tran_use_embed:
            src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        # output = torch.cat([torch.mean(output, dim=0), output[-1]], dim=1)
        # output = torch.mean(output, dim=0)
        output = torch.flatten(output.permute(1, 0, 2), start_dim=1)
        output = self.drop(self.fc1(output))
        output = self.fc2(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ConcreteDropout(nn.Module):
    def __init__(self, feature_dim, device, config, init_prob=0.5, backbone='deeplob', path=None):
        super(ConcreteDropout, self).__init__()
        self.device = device
        self.config = config
        # convolution blocks
        p_tensor = init_prob * torch.ones(feature_dim)
        self.p_logit = nn.Parameter(torch.log(p_tensor) - torch.log(1 - p_tensor), requires_grad=True)

        if backbone == 'MLP':
            self.model = MLP(feature_dim, config=config)
        else:
            self.model = deeplob(feature_dim, device, config=config)
        if path:
            self.model.load_state_dict(torch.load(path))

    def forward(self, x):
        x, z = self._concrete_dropout(x, self.p_logit)
        reg = torch.sum(z) / (x.size()[0] * x.size()[-1])
        x = self.model(x)
        if self.training:
            return x, reg
        else:
            return x

    def _concrete_dropout(self, x, p):
        p = torch.sigmoid(p)
        temp = 0.1

        unif_noise = torch.rand_like(x)

        approx = (
                torch.log(p + EPSILON)
                - torch.log(1. - p + EPSILON)
                + torch.log(unif_noise + EPSILON)
                - torch.log(1. - unif_noise + EPSILON)
        )
        drop_prob = torch.sigmoid(approx / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)

        x /= retain_prob

        return x, random_tensor


class FS(nn.Module):
    def __init__(self, feature_dim, device, config, backbone='deeplob', path=None):
        super(FS, self).__init__()
        self.device = device
        self.config = config
        # convolution blocks
        self.feature_select = nn.Parameter(torch.full([1, feature_dim], 1 / feature_dim), requires_grad=True)
        if backbone == 'MLP':
            self.model = MLP(feature_dim, config=config)
        else:
            self.model = deeplob(feature_dim, device, config=config)
        if path:
            self.model.load_state_dict(torch.load(path))

    def forward(self, x):
        x = torch.mul(x, torch.clamp(self.feature_select, 0.0, 1.0))
        x = self.model(x)
        return x
