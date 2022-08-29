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

    data_collector = pd.DataFrame(columns=('day', 'macro_f1', 'trade_time', 'winrate',
                                           'usefulrate', 'drowdown', 'return', 'hold','buy_throd','sell_throd'))
    # save_root = './checkpoints/BTC_50_LSTM/'
    # save_root = './checkpoints/BTC_50_Transformer_En/'
    # save_root = './checkpoints/BTC_50_Transformer_raw/'
    # save_root = './checkpoints/BTC_50_DeepLOB/'
    save_root = './checkpoints/BTC_ETH_14_ours_BTC/'
    # save_root = './checkpoints/BTC_ETH_14_ours_ETH/'

    for i in range(0, config.num_of_day - 9, 4):
    # for i in range(36, config.num_of_day - 9, 4):
        config.split_data = [i, i + 6, i + 10]
        print('days: ' + str(config.split_data))
        # collectors
        daily_f1 = []
        # Begin the training

        data_set = data_loader.LoadDataset(config)
        train_dic = data_set.get_crypto_data('train')
        valid_dic = data_set.get_crypto_data('val')
        train_set = data_loader.ProcessDataset(train_dic, with_label=True, config=config)
        valid_set = data_loader.ProcessDataset(valid_dic, with_label=True, config=config)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, prefetch_factor=2,
                                  num_workers=8, drop_last=True, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=config.batch_size, prefetch_factor=2,
                                  num_workers=8, drop_last=True, pin_memory=True)
        model = getattr(model_card, config.backbone)()
        print(model.model_name)
        if config.select_fun:
            print('use selection')
            model = getattr(model_card, config.select_fun_dic[config.select_fun])(model)
        # print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        # print('input shape: ', next(iter(train_loader))['input'].size())
        # print('target shape: ', next(iter(train_loader))['target'].size())

        save_path = save_root + str(config.split_data) + model.get_model_name()
        model = model.to(device)
        # if fine-tune
        # model.load(save_path)

        optimizer = Adam(model.parameters(), lr=config.lr)
        epoch = 50

        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        if config.regression:
            loss_fn = nn.MSELoss(reduction='mean').to(device)
        else:
            # loss_fn = nn.CrossEntropyLoss().to(device)
            loss_fn = DiceLoss(square_denominator=True,
                               alpha=0.01, with_logits=False,
                               index_label_position=True,
                               reduction="mean", smooth=1.0).to(device)

        # trainer.train_epoch(config, model, optimizer, device, loss_fn, train_loader,
        #                     valid_loader, epochs=epoch, scheduler=scheduler, save_path=save_path)

        model.load(save_path)
        for j in range(0, 4):
            split = [0, 6 + j, 7 + j]

            test_dic = data_set.get_crypto_data('test', split)
            test_set = data_loader.ProcessDataset(test_dic, with_label=True, config=config)
            test_loader = DataLoader(test_set, batch_size=config.batch_size, prefetch_factor=2,
                                     num_workers=8, drop_last=True, pin_memory=True)

            val_loss, f1, backtest = trainer.evaluate(model, test_loader, device, loss_fn, config)
            daily_f1.append(f1)
            data_collector = pandas.concat([data_collector, pd.DataFrame({'day': [i+j],
                                                                          'macro_f1': [f1],
                                                                          'trade_time': [backtest[0]],
                                                                          'winrate': [backtest[1]],
                                                                          'usefulrate': [backtest[2]],
                                                                          'drowdown': [backtest[3]],
                                                                          'return': [backtest[4]],
                                                                          'hold': [backtest[5]],
                                                                          'buy_throd': [backtest[6]],
                                                                          'sell_throd': [backtest[7]]})], ignore_index=True)

        test_dic = data_set.get_crypto_data('test')
        test_set = data_loader.ProcessDataset(test_dic, with_label=True, config=config)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, prefetch_factor=2,
                                 num_workers=8, drop_last=True, pin_memory=True)
        val_loss, all_f1, backtest = trainer.evaluate(model, test_loader, device, loss_fn, config)
        begin, _, end = config.split_data
        data_collector = pandas.concat([data_collector, pd.DataFrame({'day': ['all'],
                                                                      'macro_f1': [all_f1],
                                                                      'trade_time': [backtest[0]],
                                                                      'winrate': [backtest[1]],
                                                                      'usefulrate': [backtest[2]],
                                                                      'drowdown': [backtest[3]],
                                                                      'return': [backtest[4]],
                                                                      'hold': [backtest[5]],
                                                                      'buy_throd': [backtest[6]],
                                                                      'sell_throd': [backtest[7]]})], ignore_index=True)
    data_collector.to_csv(save_root+'F1_data.csv', index=False)


