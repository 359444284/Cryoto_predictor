import os

import numpy as np
import pandas as pd


class Config:
    SEED = 100
############################################################################################
#   DataSet Setting
############################################################################################
    train_ratio = 0.8
    # default save path: ./checkpoints
    # choose data set from: fi2010, BTC_50, BTC_14, ETH_14, BTC_10
    name_dataset = "fi2010"

    # load data by days
    # [train begin,    train end,    test end]
    split_data = [0, 6, 10]
    # split_data = [4, 14, 14]
    # split_data = [36, 46, 46]
    # split_data = [0, 14, 14]
    print(split_data)

    # only for regression task
    regression = False
    # only for model accept time feature
    use_time_feature = False

    # input feature_type: 'all', a list of feature in feature_dic or selected features
    # feature_type = 'all'

    # feature_type = 'list'
    # feature subsets: S_LOB, LOB, OFI, OFS, BAI, HR, MPV, AD, TCI, TVI, MPVO, PPRESS, WVPS, PD.
    feature_list = ['LOB']

    feature_type = 'selected'
    # subset selected by different model:
    # 1.  XGBoost size 25
    # 2.  WFS size 25
    # 3.  DFR size 25
    # 4.  13_common size 13 (default)
    subset_name = '13_common'

    # cloud dic
    datadict = {
        "BTC_10": "data/BTC_10/",
        "BTC_50": "data/BTC_50/",
        "BTC_14": "data/BTC_14/",
        "ETH_14": "data/ETH_14/",
        "fi2010": "data/FI_2010/"
    }
    data_path = datadict[name_dataset]
    file_name = sorted(os.listdir(data_path))
    num_of_day = len(file_name)
    # print(file_name)
    print('Total number of file: ', num_of_day)

    feature_dic = dict()
    feature_dic['SLOB'] = [i for i in range(4, 4 + 38)]
    feature_dic['LOB'] = [i for i in range(42, 42 + 40)]
    feature_dic['VFI'] = [i for i in range(82, 82 + 10)]
    feature_dic['VFS'] = [i for i in range(92, 92 + 20)]
    feature_dic['BAI'] = [i for i in range(112, 112 + 10)]
    feature_dic['HR'] = [i for i in range(122, 122 + 9)]
    feature_dic['MPV'] = [i for i in range(131, 131 + 4)]
    feature_dic['AD'] = [i for i in range(135, 135 + 2)]
    feature_dic['TCI'] = [i for i in range(137, 137 + 1)]
    feature_dic['TVI'] = [i for i in range(138, 138 + 1)]
    feature_dic['MPVO'] = [i for i in range(139, 139 + 1)]
    feature_dic['PPRESS'] = [i for i in range(140, 140 + 1)]
    feature_dic['WVPS'] = [i for i in range(141, 141 + 1)]
    feature_dic['PD'] = [i for i in range(142, 142 + 18)]

    feature_index = [0, 1, 2, 3] # use for backtesting [date, mid-price, bid1, ask1]

    if feature_type == 'all':
        for sublist in feature_dic.values():
            feature_index.extend(sublist)
    elif feature_type == 'list':
        for feature_name in feature_list:
            feature_index.extend(feature_dic[feature_name])
    else:
        if name_dataset != 'ETH_14' and name_dataset != 'BTC_14':
            if subset_name == 'WFS':
                feature_index.extend(
                    [22, 23, 102, 140, 92, 137, 82, 138, 112, 114, 113, 63, 43, 42, 134, 139, 115, 133, 62, 24, 103, 65,
                     67, 135, 29])
            elif subset_name == 'DFR':
                feature_index.extend(
                    [137, 138, 22, 23, 140, 82, 102, 92, 42, 135, 112, 43, 134, 26, 114, 113, 139, 75, 24, 32, 71, 67,
                     56, 69, 158])
            elif subset_name == 'XGBoost':
                feature_index.extend(
                    [114, 113, 112, 42, 43, 137, 138, 62, 93, 63, 115, 22, 92, 23, 102, 103, 94, 117, 141, 83, 61, 84,
                     139, 140, 116])
            else:
                feature_index.extend([102, 137, 138, 139, 140, 42, 43, 112, 113, 114, 22, 23, 92])
        else:
            feature_index.extend([102, 137, 138, 139, 140, 42, 52, 112, 113, 114, 22, 32, 92])

    if name_dataset == "fi2010":
        feature_dim = 40
    else:
        if feature_type == 'all':
            feature_dim = len([item for sublist in feature_dic.values() for item in sublist])
        else:
            feature_dim = len(feature_index) - 4

############################################################################################
#   Experiment Setting
############################################################################################

    # Choose model from LSTM DeepLOB TransformerEn
    # backbone = 'LSTM'
    # backbone = 'TransformerEn'
    backbone = 'DeepLOB'

    # whether use selection model
    # Choose between WFS DFR
    select_fun = False
    # select_fun = 'WFS'

    batch_size = 512
    if name_dataset == 'fi2010':
        lr = 0.01
    else:
        lr = 0.0005

    # Choose Loss Function Between Cross Entropy (CE) and Self-adjust Dice Loss (DSC)
    loss_fun = 'CE'
    # Alpha for DSC
    DSC_alpha = 0.2

    # the minimum learning rate will be reach.
    min_lr = lr / 20

    # backtesting setting
    trade_fee = 0.02  # unit %
    trade_delay = 1  # unit 100ms
    signal_threshold = 0.75  #

    # Choose Normalizer from: general, daily(Only available for our data set), LC-Norm
    Normalizer = 'LC-Norm'
    LC_window = None  # default is equal to input size
    assert (Normalizer in ['LC-Norm', 'general', 'daily'])

    # lockback_window is input size,
    # forecast_horizon is out put size and equal to 3 in classification tasks
    forecast_stride = 1
    training_stride = 1
    label_len = 20
    if regression:
        lockback_window = 100
        forecast_horizon = 20
    else:
        lockback_window = 100
        forecast_horizon = 3

        # choose label and horizontal
        # Label Equation 1: 160: 20 161: 50 162: 70 163: 100
        # Label Equation 2: 164: 20 165: 50 166: 70 167: 100
        feature_index.extend([161])

    if LC_window is None:
        LC_window = lockback_window

    assert (LC_window >= lockback_window)

    assert (1 <= training_stride <= lockback_window)

    # fi-2020
    if name_dataset == 'fi2010':
        forecast_horizon = 3
        forecast_stride = 1
        k = 4
    #   k = 10 20 30 50 100 horizon

    # parameter for selection models
    select_fun_dic = {
        'WFS': 'FS',
        'DFR': 'ConcreteDropout'
    }

    selection_mode = False

    if select_fun:
        if select_fun == 'VDP':
            selection_mode = True
            reg_factor = 1e-5
            anneal = 1
        elif select_fun == 'WFS':
            selection_mode = True
            reg_factor = 1e-4
            anneal = 1
        else:
            pass

    # if need a quick debug
    debug_mode = False
    debug_num = 50000

    if name_dataset == 'fi2010':
        plot_forecast = False
        backtesting = False
    else:
        plot_forecast = True
        backtesting = True

    preprocess = True
    if name_dataset == 'fi2010':
        preprocess = True

############################################################################################
#   Model Setting
############################################################################################

    # transformer
    tran_use_embed = True
    if tran_use_embed:
        tran_emb_dim = 32
    else:
        tran_emb_dim = feature_dim
    tran_layer = 2
    tran_fc_dim = tran_emb_dim * 4
    tran_num_head = 3
    tran_drop = 0.3
    use_channel_att = True
    factor = 1




