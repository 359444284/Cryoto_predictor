import os

import numpy as np
import pandas as pd


class Config:
    SEED = 100


    train_val_ratio = 0.7  # 0.7
    train_ratio = 0.8
    test_ratio = 0.3

    assert ((train_val_ratio + test_ratio) == 1)

    name_dataset = "crypto_class"
    # datadict = {
    #     "crypto": "../data/Crypto/crypt_10/",
    #     "fi2010": "../data/FI_2010/"
    # }
    # cloud dic
    datadict = {
        # "crypto": "data/crypto_class3/",
        # "crypto_class": "data/crypto_class3/",
        # "crypto_class": "data/validation/",
        "crypto_class": "data/test/",
        "fi2010": "data/FI_2010/"
    }

    data_path = datadict[name_dataset]

    num_of_day = 7

    # load data by days

    file_name = sorted(os.listdir(data_path))
    print(file_name)

    # [train begin,    train end,    test end]
    # split_data = [5, 6, 7]
    # split_data = [5, 6, 7]
    # split_data = [6, 7, 8]
    # split_data = [7, 8, 9]
    # split_data = [8, 9, 10]
    split_data = [5, 6, 10]
    # split_data = [0, 6, 7]
    # split_data = [12, 18, 19]
    # split_data = [40, 41, 48]
    # split_data = [43, 44, 50]
    # split_data = [38, 39, 44]
    # split_data = [0, 3, 4]
    print(split_data)

    # crypto data
    regression = False
    use_all_features = False
    use_time_feature = False

    # LSTM DeepLOB TransformerEn MLPMixer DeepFolio
    backbone = 'LSTM'

    # select_fun = False
    select_fun = 'FS'
    # FS VDP
    # select_fun = 'FS'

    batch_size = 512
    lr = 0.0005
    min_lr = lr / 20

    feature_dic = dict()

    feature_dic['S_LOB'] = [i for i in range(4, 4 + 38)]
    feature_dic['LOB'] = [i for i in range(42, 42 + 40)]
    feature_dic['OFI'] = [i for i in range(82, 82 + 10)]
    feature_dic['OFS'] = [i for i in range(92, 92 + 20)]
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

    # feature_dic['S_LOB'] = [i for i in range(4, 4 + 38)]
    # feature_dic['PD'] = [i for i in range(42, 42 + 36)]
    # feature_dic['BAI'] = [i for i in range(78, 78 + 10)]
    # feature_dic['HR'] = [i for i in range(88, 88 + 9)]
    # feature_dic['MPV'] = [i for i in range(97, 97 + 4)]
    # feature_dic['AD'] = [i for i in range(101, 101 + 2)]
    # feature_dic['TCI'] = [i for i in range(103, 103 + 1)]
    # feature_dic['TVI'] = [i for i in range(104, 104 + 1)]
    # feature_dic['MPVO'] = [i for i in range(105, 105 + 1)]
    # feature_dic['PPRESS'] = [i for i in range(106, 106 + 1)]
    # feature_dic['WVPS'] = [i for i in range(107, 107 + 1)]

    # feature_list = ["BAI", "HR", "MPV", "AD", 'PPRESS', 'TCI', 'TVI', 'MPVO', 'WVPS']
    feature_list = ["BAI", "MPV", "AD", 'PPRESS', 'TCI', 'TVI', 'MPVO', 'WVPS']
    # feature_list = ['TCI']

    # feature_index = [0]
    feature_index = [0, 1, 2, 3]

    # if use_all_features:
    #     for sublist in feature_dic.values():
    #         feature_index.extend(sublist)
    # else:
    #     for feature_name in feature_list:
    #         feature_index.extend(feature_dic[feature_name])
    # feature_index.extend([137, 138, 22, 23, 140, 102, 82, 92, 135, 24, 38, 28, 37, 134, 35, 36, 26, 32, 65, 30, 112, 40, 139, 124, 33, 73, 27, 31, 113, 74])
    # feature_index.extend([115, 114, 112, 113, 43, 93, 62, 102, 137, 138, 139, 92, 141, 42, 22, 63, 23, 94, 84, 95, 104, 101, 117, 83, 82, 140, 111, 134, 67, 105])
    # feature_index.extend([92, 112, 137, 102, 22, 105, 103, 114, 82, 42, 93, 20, 19, 149, 97, 80, 68, 24, 95, 81, 73, 106, 108, 78, 34, 109])
    # feature_index.extend([82, 83, 137, 22, 112, 113, 114, 138, 115, 116, 141, 118, 117, 84, 119, 120, 93, 62, 66, 70, 68, 64, 72, 76, 121, 94, 80, 21, 74, 159])
    # feature_index.extend([82, 137, 112, 113, 62, 138, 92, 115, 121, 42, 93, 84, 96, 97, 19, 110, 104, 34, 150, 156, 73, 81, 28, 71, 32, 127, 151, 101, 80, 94])
    # feature_index.extend([82, 137, 138, 92, 43, 112, 113, 62, 115, 141, 120, 117, 153, 86, 93, 83, 84, 85, 107, 102, 111, 104, 150, 156, 109, 69, 75, 106, 142, 76])
    # XGboost top 30 series
    # feature_index.extend([114, 112, 113, 43, 42, 137, 117, 116, 22, 62, 118, 138, 115, 23, 102, 93, 92, 63, 103, 139, 44, 140, 61, 83, 104, 60, 94, 84, 56, 82])
    feature_index.extend([114, 112, 113, 43, 42, 137, 117, 116, 22, 62, 118, 138, 115, 23, 102, 93, 92, 63, 103, 139, 44, 140, 61, 83, 104])

    if name_dataset == "fi2010":
        feature_dim = 40
    else:
        if use_all_features:
            feature_dim = len([item for sublist in feature_dic.values() for item in sublist])
        else:
            feature_dim = len(feature_index) - 4

    lockback_window = 100
    forecast_horizon = 20
    label_len = 20
    forecast_stride = 1
    training_stride = 1
    assert (1 <= training_stride <= lockback_window)

    if not regression:
        # feature_dim += 1
        forecast_horizon = 3
        forecast_stride = 1
        # 106: 10 107: 20 108: 50 109: 100
        # 108: 20 109: 30 110: 50 111: 100
        # feature_index.extend([110])
        feature_index.extend([161])
        # feature_index.extend([36])

    # fi-2020
    if name_dataset == 'fi2010':
        forecast_horizon = 3
        forecast_stride = 1
        k = 1
    #   10 20 30 50 100


    select_fun_dic = {
        'FS': 'FS',
        'VDP': 'ConcreteDropout'
    }

    selection_mode = False

    if select_fun:
        if select_fun == 'VDP':
            selection_mode = True
            reg_factor = 1e-4
            anneal = 1
        elif select_fun == 'FS':
            pass
        else:
            pass


    debug_mode = False
    debug_num = 50000

    plot_forecast = False
    backtesting = True

    preprocess = True
    if name_dataset == 'fi2010':
        preprocess = True



    # CNNi,l.

    # transformer
    tran_use_embed = True
    if tran_use_embed:
        tran_emb_dim = 32
    else:
        # tran_emb_dim = 32
        tran_emb_dim = feature_dim
    tran_layer = 2
    tran_fc_dim = tran_emb_dim * 4
    # tran_fc_dim = feature_dim * 4
    tran_num_head = 8
    tran_drop = 0.3

    # autoformer
    freq = 's'
    output_attention =True
    moving_avg = 25
    embed_type = 0
    enc_in = feature_dim+1
    dec_in = feature_dim+1
    c_out = 1
    d_model = 64
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = d_model * 4
    factor = 1
    dropout = 0.2
    embed = 'timeF'
    activation = 'gelu'

    # Dlinear




