import torch
from torch import nn
from .BasicModule import BasicModule


class FS(BasicModule):
    def __init__(self, model):
        super(FS, self).__init__()
        feature_dim = self.config.feature_dim
        self.model_name = 'FS'
        # convolution blocks
        self.feature_select = nn.Parameter(torch.ones([1, 1, feature_dim]), requires_grad=True)
        self.factor = 1
        self.model = model

    def forward(self, x):
        B, L, C = x.shape

        # if self.training:
        #     noise = (torch.rand_like(self.feature_select)) * max(self.feature_select)
        #     tmp_top_k = torch.topk(self.feature_select[0, 0, :], 50, largest=True).indices
        #     noise[0, 0, tmp_top_k] = 0.0
        #     selection = torch.sigmoid((self.feature_select + noise)* 4.)
        # else:
        #     selection = torch.sigmoid(self.feature_select * 4.)
        # x = torch.mul(x, selection)
        # top_k = torch.topk(selection[0, 0, :], 50).indices

        # x = x[:,:,top_k]
        x = torch.mul(x, torch.clamp(self.feature_select, 0, 1))
        x, _ = self.model(x)
        return x, _

    def save(self, name=None, last=False):
        if name is None:
            name = 'checkpoints/'
            if last:
                name += 'last_' + self.model.model_name + '_' + self.get_model_name()
            else:
                name += 'best_' + self.model.model_name + '_' + self.get_model_name()
        torch.save(self.state_dict(), name)
        return name

    def load(self, name=None, last=False):
        if name is None:
            name = 'checkpoints/'
            if last:
                name += 'last_' + self.model.model_name + '_' + self.get_model_name()
            else:
                name += 'best_' + self.model.model_name + '_' + self.get_model_name()
        self.load_state_dict(torch.load(name))