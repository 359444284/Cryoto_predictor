import torch
from torch import nn
from .BasicModule import BasicModule

class MLP(BasicModule):
    def __init__(self):
        super(MLP, self).__init__()
        self.model_name = 'mlp'

        self.fc = nn.Sequential(
            nn.Linear(self.config.feature_dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, int(self.config.forecast_horizon / self.config.forecast_stride)),
        )

    def forward(self, x):
        # print(x.size())
        x = torch.squeeze(x, 1)
        x = self.fc(x)

        return x