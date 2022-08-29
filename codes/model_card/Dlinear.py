from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .BasicModule import BasicModule

# adapted form https://github.com/cure-lab/LTSF-Linear.git

class Deeplob_CNN(nn.Module):
    def __init__(self):
        super(Deeplob_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
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
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5)),
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
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.squeeze()
        return x.permute(0, 2, 1)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Dlinear(BasicModule):
    """
    DLinear
    """
    def __init__(self):
        super(Dlinear, self).__init__()
        configs = self.config
        self.model_name = 'dlinear'
        self.seq_len = configs.lockback_window
        self.pred_len = 3

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = configs.feature_dim

        self.lstm1 = nn.LSTM(self.channels, 32, 2 ,batch_first=True)
        self.lstm2 = nn.LSTM(self.channels, 32, 2 ,batch_first=True)
        self.fc = nn.Linear(64, 3)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # x = self.feature_learning(x)
        # feature, price = x[:, :, :-1], x[:, :, -1]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output, _ = self.lstm1(seasonal_init.permute(0,2,1), None)
        trend_output, _ = self.lstm2(trend_init.permute(0,2,1), None)
        price = self.drop(torch.cat((seasonal_output[:, -1, :], trend_output[:, -1, :]), dim=1))
        # price = torch.mul(seasonal_output, trend_output)
        # x, _ = self.lstm(price, None)
        # x = x[:, -1, :]
        # x = self.dropout(x)
        x = self.fc(price)

        # # x = torch.cat([x, price], dim=1)
        # x = torch.cat([x, price], dim=1)
        # x = self.fc2(x).unsqueeze(dim=-1)

        return x, None
