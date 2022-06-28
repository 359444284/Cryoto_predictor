import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW, lr_scheduler, Adam
from warmup_scheduler import GradualWarmupScheduler

import config
import data_loader
import models
import trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    config = config.Config()
    set_seed(config.SEED)
    data_set = data_loader.LoadDataset(config)
    print(config.feature_dim)

    if config.name_dataset == 'crypto':
        train_x, train_y = data_set.get_crypto_data('train')
        valid_x, valid_y = data_set.get_crypto_data('val')
        print(len(train_x), len(train_y))
        print(len(valid_x), len(valid_y))
        train_set = data_loader.ProcessDataset(train_x, train_y, with_label=True, config=config)
        valid_set = data_loader.ProcessDataset(valid_x, valid_y, with_label=True, config=config)

    else:
        train_x, train_y = data_set.get_FI_data('train')
        valid_x, valid_y = data_set.get_FI_data('val')
        print(len(train_x), len(train_y))
        print(len(valid_x), len(valid_y))
        train_set = data_loader.ProcessDataset(train_x, train_y, with_label=True, config=config, regression=False)
        valid_set = data_loader.ProcessDataset(valid_x, valid_y, with_label=True, config=config, regression=False)

    # target = train_set.targets.tolist()
    #
    # target = [i for i in target]
    # class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    # print(class_sample_count)
    # # apply weight sampling
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[int(t)] for t in target])
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weight = samples_weight.double()
    # sampler = WeightedRandomSampler(samples_weight, int(len(train_set)*1.5), replacement=True)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=0, shuffle=True, drop_last=True)
    # train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=0, sampler=sampler ,shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, num_workers=0, drop_last=True)
    # model = models.CNN(input_dim=data_set.feature_dim, config=config)
    # model = models.MLP(config=config)
    # model = models.deeplob(device, config)
    # model = models.MLPMixer(config)
    model = models.LSTM(device, config)
    # model = models.TransformerModel(device, config)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('input shape: ', next(iter(train_loader))['input'].size())
    print('target shape: ', next(iter(train_loader))['target'].size())
    model = model.to(device)

    # model.load_state_dict(torch.load('./baiandsing_deeplob.bin'))
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    # optimizer = Adam(model.parameters(), lr=config.lr)
    epoch = 100

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=2, total_epoch=5, after_scheduler=scheduler)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=1, after_scheduler=scheduler)

    if config.name_dataset == 'crypto':
        loss_fn = nn.SmoothL1Loss(reduction='mean').to(device)
        # loss_fn = nn.MSELoss(reduction='mean').to(device)
    else:
        loss_fn = nn.CrossEntropyLoss()

    trainer.train_epoch(config, model, optimizer, device, loss_fn, train_loader,
                        valid_loader, epochs=epoch, scheduler=scheduler)
