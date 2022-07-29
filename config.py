import numpy as np


class Config:
    SEED = 100

    train_val_ratio = 0.7  # 0.7
    train_ratio = 0.75
    test_ratio = 0.3
    # train_val_ratio = 0.8  # 0.7
    # train_ratio = 0.875
    # test_ratio = 0.2
    assert ((train_val_ratio + test_ratio) == 1)

    name_dataset = "crypto"
    datadict = {
        "crypto": "data/newfea.csv",
        # "crypto": "data/dataset_4_29_to_5_5.csv",
        "fi2010": "data/FI_2010/"
    }

    data_path = datadict[name_dataset]

    num_of_day = 7

    # crypto data

    use_all_features = False
    feature_dic = dict()
    # feature_dic['OFI'] = [i for i in range(2, 2 + 10)]
    # feature_dic['MPVO'] = [i for i in range(12, 12 + 1)]
    # feature_dic['BAI'] = [i for i in range(13, 13 + 10)]
    # feature_dic['PD'] = [i for i in range(23, 23 + 36)]
    # feature_dic['TCI'] = [i for i in range(59, 59 + 1)]
    # feature_dic['TVI'] = [i for i in range(60, 60 + 1)]
    # feature_dic['WVPS'] = [i for i in range(61, 61 + 1)]
    # feature_dic['HR'] = [i for i in range(62, 62 + 9)]
    # feature_dic['AD'] = [i for i in range(71, 71 + 2)]
    # feature_dic['LOB'] = [i for i in range(73, 73 + 40)]
    # feature_dic['MPV'] = [i for i in range(113, 113 + 4)]
    # feature_dic['OFS'] = [i for i in range(117, 117 + 20)]
    # feature_dic['PPRESS'] = [i for i in range(137, 137 + 1)]

    feature_dic['S_LOB'] = [i for i in range(2, 2 + 38)]
    feature_dic['PD'] = [i for i in range(40, 40 + 36)]
    feature_dic['BAI'] = [i for i in range(76, 76 + 10)]
    feature_dic['HR'] = [i for i in range(86, 86 + 9)]
    feature_dic['MPV'] = [i for i in range(95, 95 + 4)]
    feature_dic['AD'] = [i for i in range(99, 99 + 2)]
    feature_dic['TCI'] = [i for i in range(101, 101 + 1)]
    feature_dic['TVI'] = [i for i in range(102, 102 + 1)]
    feature_dic['MPVO'] = [i for i in range(103, 103 + 1)]
    feature_dic['PPRESS'] = [i for i in range(104, 104 + 1)]
    feature_dic['WVPS'] = [i for i in range(105, 105 + 1)]

    feature_list = ["BAI", "MPV", "AD", 'PPRESS', 'TCI', 'TVI', 'MPVO', 'WVPS']  # dum = 21 6 32 lr 0.001  (3,2), (3,2), 4 512
    # feature_list = ["MPV", "AD", 'PPRESS', 'TCI', 'TVI', 'MPVO', 'WVPS'] # dim = 11 lr=0.0005 7, 0.2, (3,2), (3,2), 2  512
    # feature_list = ['BAI', 'OFI', 'PPRESS', 'TCI', 'TVI', 'MPVO', 'WVPS'] # 7 0.2637
    # all feature batch_size = 128 lr 0.001 6 32.5 dim = 136 (5,4) (5,4) 8
    # feature_list = ['LOB'] # 5 23
    # feature_list = ['OFS']

    feature_index = [0, 1]
    # FS_selectby_MLP_Drop = [2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35,
    #                         36, 39, 40, 41, 42, 43, 44, 46, 47, 48, 51, 52, 53, 55, 56, 57, 59, 60, 61, 71, 72, 93, 94,
    #                         115, 116, 117, 127, 137]
    # FS_selectby_deeplob_Drop = [3, 4, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 27, 28, 31, 35, 36, 39, 40, 43, 44, 47,
    #                             48, 51, 52, 55, 56, 58, 59, 60, 61, 62, 63, 64, 67, 68, 75, 91, 95, 99, 103, 107, 111,
    #                             115, 119, 123, 124, 127, 128, 131]
    for feature_name in feature_list:
        feature_index.extend(feature_dic[feature_name])
    # feature_index.extend(FS_selectby_deeplob_Drop)

    if name_dataset == "crypto":
        if use_all_features:
            feature_dim = len([item for sublist in feature_dic.values() for item in sublist])
        else:
            feature_dim = len(feature_index) - 2
    else:
        feature_dim = 40

    lockback_window = 100
    forecast_horizon = 20
    forecast_stride = 1
    # assert (lockback_window >= forecast_horizon)

    training_stride = 1
    assert (1 <= training_stride <= lockback_window)

    # fi-2020
    if name_dataset == 'fi2010':
        forecast_horizon = 3
        forecast_stride = 1
        k = 4

    selection_mode = False
    reg_factor = 1e-6
    anneal = 30

    debug_mode = False
    debug_num = 30000

    preprocess = False
    if name_dataset == 'fi2010':
        preprocess = True

    batch_size = 512
    # lr = 0.00005 # MLP
    lr = 0.001
    min_lr = lr / 20

    # CNN

    # transformer
    tran_use_embed = False
    if tran_use_embed:
        tran_emb_dim = 8
    else:
        tran_emb_dim = feature_dim
    tran_layer = 2
    tran_fc_dim = 128
    tran_num_head = 3
    tran_drop = 0.0
