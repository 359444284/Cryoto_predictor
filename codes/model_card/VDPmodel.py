import torch
from torch import nn
from .BasicModule import BasicModule

EPSILON = 1e-6

class ConcreteDropout(BasicModule):
    def __init__(self, model, init_prob=0.5):
        super(ConcreteDropout, self).__init__()
        self.model_name = 'ConcreteDropout'
        # convolution blocks
        p_tensor =  torch.ones(self.config.feature_dim) * init_prob
        self.p_logit = nn.Parameter(torch.log(p_tensor) - torch.log(1 - p_tensor), requires_grad=True)

        self.model = model

    def forward(self, x):

        if self.training:
            x, z = self._concrete_dropout(x, self.p_logit)
            reg = torch.sum(z) / (x.size()[0] * x.size()[-1])
            x, _ = self.model(x)
            return x, reg
        else:
            # x, z = self._concrete_dropout(x, self.p_logit)
            x, _ = self.model(x)
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
