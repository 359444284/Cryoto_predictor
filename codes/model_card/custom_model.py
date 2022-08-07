import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from einops.layers.torch import Rearrange
import CNN
EPSILON = np.finfo(float).eps




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

