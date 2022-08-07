import torch
from torch import nn
from .BasicModule import BasicModule

class CNN(BasicModule):
    def __init__(self):
        super().__init__()
        # convolution blocks
        self.model_name = 'cnn'
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            #             nn.Tanh(),
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
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.Tanh(),
            nn.BatchNorm2d(32),
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

        self.fc1 = nn.Linear(32 * 100, int(self.config.forecast_horizon / self.config.forecast_stride))

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        x = torch.unsqueeze(x, 1)
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x
