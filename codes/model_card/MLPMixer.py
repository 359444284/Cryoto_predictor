# adapted from https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py
import torch
from torch import nn
from einops.layers.torch import Rearrange
from .BasicModule import BasicModule


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

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.05):
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

class MLPMixer(BasicModule):

    def __init__(self, dim=40, patch_size=4, depth=2, token_dim=64, channel_dim=40*4):
        super().__init__()
        assert self.config.lockback_window % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.model_name = 'mlp_mixer'
        self.num_patch =  (self.config.lockback_window//patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(self.config.feature_dim, dim, (patch_size, 1), (patch_size, 1)),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, self.config.forecast_horizon//self.config.forecast_stride)
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