import torch
from torch import nn
from .BasicModule import BasicModule
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, (3, 1), (1, 1), padding='same'),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm2d(outchannel),
                nn.Conv2d(outchannel, outchannel, (3, 1), (1, 1), padding='same'),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm2d(outchannel),
                nn.Conv2d(outchannel, outchannel, (3, 1), (1, 1), padding='same'),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class DeepFolio(BasicModule):
    def __init__(self, cnn_dim=16, incep_dim=32, lstm_dim=64):
        super().__init__()
        self.model_name = 'deepfolio'
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn_dim, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(cnn_dim),
        )
        self.layer1 = self._make_layer(cnn_dim, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(cnn_dim),
        )
        self.layer2 = self._make_layer(cnn_dim, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(cnn_dim),
        )
        self.layer3 = self._make_layer(cnn_dim, 2)
        # self.fitting = nn.AdaptiveAvgPool2d((None, 1))
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
            nn.Conv2d(in_channels=incep_dim, out_channels=incep_dim, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(incep_dim),
            nn.Conv2d(in_channels=incep_dim, out_channels=incep_dim, kernel_size=(3, 1), padding='same'),
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

    def _make_layer(self, inchannel, block_num):

        layers = []

        for i in range(1, block_num):
            layers.append(ResidualBlock(inchannel, inchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64).to(self.device)
        c0 = torch.zeros(1, x.size(0), 64).to(self.device)
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.layer3(x)
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

        return x, None
