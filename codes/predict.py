import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler, Adam
import seaborn as sns
import config
import data_loader
import trainer
import model_card

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':

    config = config.Config()
    data_set = data_loader.LoadDataset(config)
    if config.name_dataset == 'fi2010':
        valid_dic = data_set.get_FI_data('val')
        test_dic = data_set.get_FI_data('test')
        valid_set = data_loader.ProcessDataset(valid_dic, with_label=True, config=config)
        test_set = data_loader.ProcessDataset(test_dic, with_label=True, config=config)
    else:
        valid_dic = data_set.get_crypto_data('val')
        test_dic = data_set.get_crypto_data('test')
        valid_set = data_loader.ProcessDataset(valid_dic, with_label=True, config=config)
        test_set = data_loader.ProcessDataset(test_dic, with_label=True, config=config)

    print(len(valid_set))
    print(len(test_set))
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, drop_last=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, drop_last=True, num_workers=8, pin_memory=True)
    model = getattr(model_card, config.backbone)()
    if config.select_fun:
        print('use selection method')
        model = getattr(model_card, config.select_fun_dic[config.select_fun])(model)
    print(model.get_model_name())

    #  load the pre-trained model
    model.load()
    # model.load(name='checkpoints/FI-2010/best_deeplob_LOB_100_100_.pt')

    model = model.to(device)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    optimizer = Adam(model.parameters(), lr=config.lr)

    if config.regression:
        loss_fn = nn.MSELoss(reduction='mean').to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)
    val_loss, f1, backtest = trainer.evaluate(model, test_loader, device, loss_fn, config)


    for name, param in model.named_parameters():
        if name == 'p_logit':
                print(1- torch.sigmoid(param))

    for name, param in model.named_parameters():
        if name == 'feature_select':
                print(param)
    # for i in range(0, 4):
    #     split = [0, 6 + i, 7 + i]
    #     test_dic = data_set.get_crypto_data('test', split)
    #     test_set = data_loader.ProcessDataset(test_dic, with_label=True, config=config)
    #     test_loader = DataLoader(test_set, batch_size=config.batch_size, prefetch_factor=2,
    #                              num_workers=8, drop_last=True, pin_memory=True)
    #
    #     val_loss, r2, backtest = trainer.evaluate(model, test_loader, device, loss_fn, config)


