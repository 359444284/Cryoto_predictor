import math

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .BasicModule import BasicModule


# adapted from <DeepLOB: Deep Convolutional Neural Networks
# for Limit Order Books>
# https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/blob/master/jupyter_pytorch/run_train_pytorch.ipynb
class Learned_Aggregation_Layer(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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

# 0.7580
class DeepLOB(BasicModule):
    def __init__(self, cnn_dim=32, incep_dim=64, lstm_dim=64):
        super().__init__()
        # convolution blocks
        self.model_name = 'deeplob'

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn_dim, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Tanh(),
            nn.BatchNorm2d(cnn_dim),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Tanh(),
            nn.BatchNorm2d(cnn_dim),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Tanh(),
            nn.BatchNorm2d(cnn_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(1, 2), stride=(1, 2)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Tanh(),
            nn.BatchNorm2d(cnn_dim),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(4, 1), padding='same'),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Tanh(),
            nn.BatchNorm2d(cnn_dim),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(4, 1), padding='same'),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Tanh(),
            nn.BatchNorm2d(cnn_dim)
        )
        # self.conv3 = nn.Sequential(
        #     Learned_Aggregation_Layer(32),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.BatchNorm2d(32),
        # )
        # last_dim = self.config.feature_dim -
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(1, self.config.feature_dim//4)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(cnn_dim),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(cnn_dim),
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(cnn_dim),
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_dim, out_channels=incep_dim, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(incep_dim),
            nn.Conv2d(in_channels=incep_dim, out_channels=incep_dim, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(incep_dim),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_dim, out_channels=incep_dim, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(incep_dim),
            nn.Conv2d(in_channels=incep_dim, out_channels=incep_dim, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(incep_dim),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=cnn_dim, out_channels=incep_dim, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(incep_dim),
        )
        # lstm layers
        self.lstm = nn.LSTM(input_size=incep_dim*3, hidden_size=lstm_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(lstm_dim, int(self.config.forecast_horizon / self.config.forecast_stride))
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x, None)
        x = x[:, -1, :]
        x = self.fc1(x)

        return x, None