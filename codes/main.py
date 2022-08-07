import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW, lr_scheduler, Adam
# from warmup_scheduler import GradualWarmupScheduler
from loss_funs import DiceLoss

import config
import data_loader
import trainer
import model_card

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
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1.'


if __name__ == '__main__':
    config = config.Config()
    set_seed(config.SEED)
    data_set = data_loader.LoadDataset(config)

    print(config.feature_dim)

    if config.name_dataset == 'fi2010':
        train_dic = data_set.get_FI_data('train')
        valid_dic = data_set.get_FI_data('val')
        train_set = data_loader.ProcessDataset(train_dic, with_label=True, config=config)
        valid_set = data_loader.ProcessDataset(valid_dic, with_label=True, config=config)
    else:
        train_dic = data_set.get_crypto_data('train')
        valid_dic = data_set.get_crypto_data('val')
        train_set = data_loader.ProcessDataset(train_dic, with_label=True, config=config)
        valid_set = data_loader.ProcessDataset(valid_dic, with_label=True, config=config)

    if not config.regression:
        target = train_set.targets.tolist()

        target = [i for i in target]
        #
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        #
        print(class_sample_count)
    # # apply weight sampling
    # weight = 1. / class_sample_count
    # weight[0] = weight[0] * 1.2
    # samples_weight = np.array([weight[int(t)] for t in target])
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weight = samples_weight.double()
    # sample_num = int(sum(class_sample_count[1:]))
    # sample_num = int(sample_num * 1.7)
    # print(sample_num)
    # sampler = WeightedRandomSampler(samples_weight, sample_num, replacement=True)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, prefetch_factor=2,
                              num_workers=8, drop_last=True, pin_memory=True, shuffle=True)
    # train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=0, sampler=sampler ,shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, prefetch_factor=2,
                              num_workers=8, drop_last=True, pin_memory=True)

    model = getattr(model_card, config.backbone)()
    print(model.model_name)
    # model.load()
    if config.select_fun:
        print('use selection')
        model = getattr(model_card, config.select_fun_dic[config.select_fun])(model)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('input shape: ', next(iter(train_loader))['input'].size())
    print('target shape: ', next(iter(train_loader))['target'].size())

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=config.lr)
    epoch = 100

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if config.name_dataset == 'crypto':
        # loss_fn = nn.SmoothL1Loss(reduction='mean').to(device)
        loss_fn = nn.MSELoss(reduction='mean').to(device)
    else:
        # loss_fn = nn.CrossEntropyLoss().to(device)
        # loss_fn = DiceLoss(square_denominator=True,
        #             alpha=0.01, with_logits=False,
        #             index_label_position=True,
        #             reduction="mean", smooth=1.0).to(device)
        # loss_fn = DiceLoss(square_denominator=True,
        #                    alpha=0.5, with_logits=False,
        #                    index_label_position=True,
        #                    reduction="mean", smooth=1.0).to(device)
        loss_fn = DiceLoss(square_denominator=True,
                           alpha=0.01, with_logits=False,
                           index_label_position=True,
                           reduction="mean", smooth=1.0).to(device)

    trainer.train_epoch(config, model, optimizer, device, loss_fn, train_loader,
                        valid_loader, epochs=epoch, scheduler=scheduler)