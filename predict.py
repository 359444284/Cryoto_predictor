import os
import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

import config
import data_loader
import models
import trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':

    config = config.Config()
    data_set = data_loader.LoadDataset(config)
    if config.name_dataset == 'crypto':
        test_x, test_y = data_set.get_crypto_data('test')
        train_x, train_y = data_set.get_crypto_data('train')
        valid_x, valid_y = data_set.get_crypto_data('val')
        train_set = data_loader.ProcessDataset(train_x, train_y, with_label=True, config=config)
        valid_set = data_loader.ProcessDataset(valid_x, valid_y, with_label=True, config=config)
        test_set = data_loader.ProcessDataset(test_x, test_y, with_label=True, config=config)
    else:
        train_x, train_y = data_set.get_FI_data('train')
        valid_x, valid_y = data_set.get_FI_data('val')
        test_x, test_y = data_set.get_FI_data('test')
        train_set = data_loader.ProcessDataset(train_x, train_y, with_label=True, config=config, regression=False)
        valid_set = data_loader.ProcessDataset(valid_x, valid_y, with_label=True, config=config, regression=False)
        test_set = data_loader.ProcessDataset(test_x, test_y, with_label=True, config=config, regression=False)

    print(len(train_set))
    print(len(valid_set))
    print(len(test_set))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, drop_last=True, num_workers=0, pin_memory=True)
    # print(config.feature_dim)
    # model = models.MLPMixer(config=config)
    # model = models.deeplob(device, config=config)
    model = models.LSTM(device, config)
    # model = models.TransformerModel(device, config)
    model.load_state_dict(torch.load('./best_model_state.bin'))
    model = model.to(device)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    if config.name_dataset == 'crypto':
        loss_fn = nn.MSELoss(reduction='mean').to(device)
    else:
        loss_fn = nn.CrossEntropyLoss()

    val_loss, r2 = trainer.evaluate(model, test_loader, device, loss_fn, config)
    # #
    if config.name_dataset == 'crypto':
        fig = plt.figure()
        plt.plot([i for i in range(1, config.forecast_horizon + 1, config.forecast_stride)],
                 np.clip(r2, -0.2, 1), 'o-', color='g')
        fig.suptitle("test set: " + str(max(r2)))
        plt.show()
    else:
        print(r2)

    # model = models.FS(config.feature_dim, device, config=config, path='allfea_deeplob_with_nothing.bin')
    # model = models.FS(config.feature_dim, device, config=config, backbone=None, path='allfea_deeplob_with_nothing.bin')
    # model = models.ConcreteDropout(config.feature_dim, device, config=config, backbone=None, path='allfea_deeplob_with_nothing.bin')
    # model.load_state_dict(torch.load('./2.1.bin'))
    # model = model.to(device)
    # optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    # loss_fn = nn.MSELoss(reduction='sum').to(device)
    # feature_weight = torch.sigmoid(torch.load('./dropFS_deeplob.bin')['p_logit'].detach())
    # epoch = 50

    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.5, total_epoch=5, after_scheduler=scheduler)
    #
    # trainer.train_epoch(config, model, optimizer, device, loss_fn, train_loader,
    #                     valid_loader, epochs=epoch, scheduler=scheduler_warmup)
    # print(feature_weight)

    # feature selection
    # for name, param in model.named_parameters():
    #     if name == 'feature_select':
    #         param.requires_grad = False
    #         mask = torch.ones(feature_weight.size()[0])
    #         mask = mask.to(device)
    #         param.data = mask
    # val_loss, r2 = trainer.evaluate(config, model, valid_loader, device, loss_fn)
    # fig = plt.figure()
    # plt.plot([i for i in range(1, config.forecast_horizon + 1, config.forecast_stride)],
    #          np.clip(r2, -0.2, 1), 'o-', color='g')
    # fig.suptitle("test set: " + str(max(r2)))
    # plt.show()
    # mean_score = []
    # max_score = []
    # ran = range(1, 130, 5)
    # print(torch.topk(feature_weight, 51, largest=False).indices)
    # for i in ran:
    #     for name, param in model.named_parameters():
    #         if name == 'feature_select':
    #             param.requires_grad = False
    #             mask = torch.full((1, feature_weight.size()[0]), 0.3)
    #             mask[0][torch.topk(feature_weight, i, largest=False).indices] = 1.0
    #             mask = mask.to(device)
    #             param.data = mask
    #
    #     val_loss, r2 = trainer.evaluate(config, model, valid_loader, device, loss_fn)
    #     #
    #     print(max(r2))
    #     max_score.append(max(r2))
    #     mean_score.append(np.mean(r2))
    #
    # fig = plt.figure()
    # plt.plot(ran,
    #          mean_score, 'o-', color='g')
    # fig.suptitle("mean r2")
    # plt.show()
    #
    # fig = plt.figure()
    # plt.plot(ran,
    #          max_score, 'o-', color='g')
    # fig.suptitle("max r2")
    # plt.show()
