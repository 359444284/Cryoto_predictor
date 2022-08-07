from datetime import time

import torch
import torch.nn as nn
import config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.device = device
        self.config = config.Config()
        self.model_name = str(type(self))

    def load(self, name=None, last=False):
        if name is None:
            name = 'checkpoints/'
            if last:
                name += 'last_' + self.get_model_name()
            else:
                name += 'best_' + self.get_model_name()
        self.load_state_dict(torch.load(name))

    def save(self, name=None, last=False):
        if name is None:
            name = 'checkpoints/'
            if last:
                name += 'last_' + self.get_model_name()
            else:
                name += 'best_' + self.get_model_name()
        torch.save(self.state_dict(), name)
        return name

    def get_model_name(self):
        name = self.model_name + '_'
        if self.config.use_all_features:
            name += 'allfea_'
        else:
            name += '_'.join(self.config.feature_list) + '_'

        name += str(self.config.lockback_window) + '_' \
                  + str(self.config.forecast_horizon) + '_'
        name = name + '.pt'
        return name
