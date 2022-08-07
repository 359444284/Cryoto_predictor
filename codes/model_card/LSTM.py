import torch
from torch import nn
from .BasicModule import BasicModule
from .layers.SelfAttention_Family import AttentionLayer, FullAttention
from einops import rearrange

from .layers.Transformer_EncDec import ChannelProcessing


class LSTM(BasicModule):
    def __init__(self,hidden_size=64, num_layer=2):
        super(LSTM, self).__init__()
        self.model_name = 'lstm'
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=self.config.feature_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layer,
                            batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.config.forecast_horizon / self.config.forecast_stride))
        )

    def forward(self, x):
        # h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(self.device)

        x, _ = self.lstm(x, None)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x, None
