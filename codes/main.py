import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW, lr_scheduler, Adam
# from warmup_scheduler import GradualWarmupScheduler
from loss_funs import DiceLoss
import seaborn as sns
import config
import trainer
import model_card
from data_provider import datasets, data_loader

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
    # reproduce ability , delete to get a faster training speed
    set_seed(config.SEED)

    data_set = data_loader.LoadDataset(config)

    if config.name_dataset == 'fi2010':
        train_dic = data_set.get_FI_data('train')
        valid_dic = data_set.get_FI_data('val')
        train_set = datasets.get_dataset(train_dic, with_label=True, config=config)
        valid_set = datasets.get_dataset(valid_dic, with_label=True, config=config)
    else:
        train_dic = data_set.get_crypto_data('train')
        valid_dic = data_set.get_crypto_data('val')
        train_set = datasets.get_dataset(train_dic, with_label=True, config=config)
        valid_set = datasets.get_dataset(valid_dic, with_label=True, config=config)


    if not config.regression:
        target = train_set.targets.tolist()
        target = [int(i) for i in target]
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        class_sample_count = class_sample_count/sum(class_sample_count)
        print(class_sample_count)
        # data = pandas.DataFrame({'Buy': [class_sample_count[-1]], 'Hold': [class_sample_count[0]], 'Sell':[class_sample_count[1]]})
        # sns.barplot(data=data)
        # plt.show()

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
                              num_workers=0, drop_last=True, pin_memory=False, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, prefetch_factor=2,
                              num_workers=0, drop_last=True, pin_memory=False)

    model = getattr(model_card, config.backbone, "LSTM")()

    print(model.model_name)
    if config.select_fun:
        print('use selection')
        model = getattr(model_card, config.select_fun_dic[config.select_fun])(model)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('input shape: ', next(iter(train_loader))['input'].size())
    print('target shape: ', next(iter(train_loader))['target'].size())

    model = model.to(device)
    # model.load()
    optimizer = Adam(model.parameters(), lr=config.lr)
    epoch = 50

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    if config.regression:
        # loss_fn = nn.SmoothL1Loss(reduction='mean').to(device)
        loss_fn = nn.MSELoss(reduction='mean').to(device)
    else:
        if config.loss_fun == 'CE':
            print('Using Cross Entropy')
            loss_fn = nn.CrossEntropyLoss().to(device)
        else:
            print('Using Self-adjust Dice Loss')
            loss_fn = DiceLoss(square_denominator=True,
                    alpha=config.DSC_alpha, with_logits=False,
                    index_label_position=True,
                    reduction="mean", smooth=1.0).to(device)

    trainer.train_epoch(config, model, optimizer, device, loss_fn, train_loader,
                        valid_loader, epochs=epoch, scheduler=scheduler)
    model.load(last=False)

    if config.name_dataset == 'fi2010':
        test_dic = data_set.get_FI_data('test')
        test_set = datasets.get_dataset(test_dic, with_label=True, config=config)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, prefetch_factor=2,
                                 num_workers=8, drop_last=True, pin_memory=True)
        val_loss, f1, _ = trainer.evaluate(model, test_loader, device, loss_fn, config)
        print(f1)
    else:
        for i in range(0, 4):
            split = [0, 6+i, 7+i]
            test_dic = data_set.get_crypto_data('test', split)
            test_set = datasets.get_dataset(test_dic, with_label=True, config=config)
            test_loader = DataLoader(test_set, batch_size=config.batch_size, prefetch_factor=2,
                                     num_workers=8, drop_last=True, pin_memory=True)

            val_loss, f1, _ = trainer.evaluate(model, test_loader, device, loss_fn, config)
            print(f1)
        test_dic = data_set.get_crypto_data('test')
        test_set = datasets.get_dataset(test_dic, with_label=True, config=config)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, prefetch_factor=2,
                                 num_workers=8, drop_last=True, pin_memory=True)
        val_loss, f1, _ = trainer.evaluate(model, test_loader, device, loss_fn, config)
        print(f1)