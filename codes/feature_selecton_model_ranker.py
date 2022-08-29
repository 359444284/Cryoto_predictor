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
    model.load()

    if config.select_fun:
        model = getattr(model_card, 'FS')(model)
    model = model.to(device)
    for name, param in model.named_parameters():
        print(name)
    optimizer = Adam(model.parameters(), lr=config.lr)
    if config.regression:
        loss_fn = nn.MSELoss(reduction='mean').to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # #  FS
    # feature_weight = torch.tensor([0.7074, 0.7682, 0.7277, 0.6754, 0.7088, 0.6307, 0.6931, 0.6054,
    #       0.6178, 0.6082, 0.6201, 0.5751, 0.6279, 0.6418, 0.5662, 0.6080,
    #       0.6872, 0.6872, 1.2426, 1.2020, 0.8481, 0.8055, 0.7969, 0.6835,
    #       0.7825, 0.8355, 0.8283, 0.7911, 0.7775, 0.8125, 0.8014, 0.7924,
    #       0.8291, 0.8147, 0.7665, 0.6787, 0.7424, 0.7689, 0.8783, 0.9029,
    #       0.6980, 0.6557, 0.6726, 0.6547, 0.6569, 0.6545, 0.7787, 0.6336,
    #       0.7020, 0.6581, 0.5974, 0.6311, 0.6773, 0.7128, 0.7438, 0.7273,
    #       0.6493, 0.7403, 0.8584, 0.9066, 0.7466, 0.8421, 0.7446, 0.8403,
    #       0.7344, 0.7195, 0.7496, 0.7664, 0.7650, 0.7332, 0.5748, 0.7384,
    #       0.8048, 0.7588, 0.7531, 0.7549, 0.6619, 0.7511, 1.1114, 0.7868,
    #       0.7879, 0.7678, 0.6513, 0.6378, 0.6369, 0.5053, 0.6299, 0.4778,
    #       1.1357, 0.8221, 0.8108, 0.6976, 0.6116, 0.5923, 0.5915, 0.6599,
    #       0.6042, 0.6993, 1.1859, 0.8450, 0.7908, 0.7274, 0.7863, 0.6291,
    #       0.5754, 0.7087, 0.7305, 0.7461, 1.0215, 0.9301, 0.9391, 0.8664,
    #       0.6727, 0.5939, 0.7027, 0.7775, 0.7568, 0.7396, 0.7038, 0.7883,
    #       0.6540, 0.6543, 0.7483, 0.6016, 0.6190, 0.6949, 0.4668, 0.6915,
    #       0.6840, 0.8615, 0.8752, 0.8378, 0.7863, 1.1280, 1.1084, 0.8690,
    #       1.1462, 0.7744, 0.7755, 0.6291, 0.6471, 0.5696, 0.6019, 0.6888,
    #       0.7253, 0.6799, 0.6096, 0.6969, 0.6598, 0.6581, 0.6378, 0.6557,
    #       0.6721, 0.5568, 0.6853, 0.7158])
    # VDP2
    # feature_weight = torch.tensor([0.5906, 0.5864, 0.5923, 0.5561, 0.5835, 0.5856, 0.5651, 0.5665, 0.5778,
    #     0.5548, 0.5859, 0.5451, 0.5786, 0.5614, 0.5741, 0.5615, 0.5688, 0.5773,
    #     0.7695, 0.7537, 0.6054, 0.6001, 0.6085, 0.5833, 0.5981, 0.5877, 0.5793,
    #     0.5929, 0.6050, 0.5977, 0.5812, 0.5912, 0.5674, 0.5808, 0.5922, 0.5806,
    #     0.5562, 0.5865, 0.6381, 0.6156, 0.5687, 0.5821, 0.5950, 0.5810, 0.5782,
    #     0.5763, 0.5660, 0.5575, 0.5353, 0.5704, 0.5624, 0.5857, 0.6023, 0.5504,
    #     0.5591, 0.5641, 0.5603, 0.5817, 0.5930, 0.5774, 0.5603, 0.5938, 0.5579,
    #     0.6036, 0.5792, 0.6009, 0.5777, 0.6049, 0.5870, 0.5786, 0.5804, 0.6058,
    #     0.5700, 0.5739, 0.5536, 0.5750, 0.5810, 0.5755, 0.7134, 0.5778, 0.5970,
    #     0.5657, 0.5603, 0.5676, 0.5517, 0.5653, 0.5674, 0.5767, 0.6727, 0.5692,
    #     0.5706, 0.5573, 0.5540, 0.5527, 0.5746, 0.5648, 0.5553, 0.5701, 0.6765,
    #     0.5672, 0.5640, 0.5700, 0.5621, 0.5724, 0.5718, 0.5525, 0.5675, 0.5683,
    #     0.6301, 0.6068, 0.6076, 0.5944, 0.5833, 0.5850, 0.5475, 0.5642, 0.5979,
    #     0.5615, 0.5689, 0.5556, 0.5816, 0.5414, 0.5462, 0.5533, 0.5655, 0.5554,
    #     0.5539, 0.5752, 0.5480, 0.5924, 0.6114, 0.6379, 0.5730, 0.7983, 0.7780,
    #     0.6064, 0.7341, 0.5828, 0.5845, 0.5512, 0.5868, 0.5516, 0.5810, 0.5784,
    #     0.5923, 0.5975, 0.5713, 0.5667, 0.5848, 0.5683, 0.5783, 0.5492, 0.5728,
    #     0.5747, 0.6002, 0.5549])

    # XGBOOST
    feature_weight = torch.tensor([0.00307308, 0.00219175, 0.00222428, 0.00206057, 0.00213096, 0.00235912, 0.00175697, 0.00183962, 0.00243541,
         0.00218106, 0.00220243, 0.00215446, 0.0018324, 0.00192439, 0.00260901, 0.00229553, 0.00215351, 0.00247028,
         0.01765554, 0.01423167, 0.00195414, 0.0019417, 0.00190619, 0.00203582, 0.00171267, 0.00166206, 0.00194251,
         0.00243371, 0.00169414, 0.00197938, 0.00171387, 0.00166885, 0.00178335, 0.0018351, 0.00190884, 0.0014969,
         0.00198518, 0.00152204, 0.05044684, 0.05609723, 0.00769489, 0.00283361, 0.00390799, 0.00278641, 0.00238566,
         0.00238346, 0.00246087, 0.00203306, 0.00262252, 0.00288963, 0.0029829, 0.00317745, 0.00545727, 0.00206057,
         0.00311994, 0.00399826, 0.00603879, 0.00697224, 0.01716152, 0.01105127, 0.00310578, 0.00285695, 0.00300768,
         0.00387422, 0.00358696, 0.00283522, 0.00299954, 0.00302657, 0.00355144, 0.00295965, 0.00365416, 0.00332215,
         0.00312583, 0.00258763, 0.00321038, 0.00242006, 0.00344352, 0.00302744, 0.00528374, 0.00694191, 0.00572423,
         0.00314403, 0.00276517, 0.002984, 0.0024025, 0.00204968, 0.00260858, 0.0014968, 0.01124795, 0.01193252,
         0.00589066, 0.00525919, 0.00299138, 0.00341358, 0.00218256, 0.00345053, 0.0025149, 0.00286558, 0.01225451,
         0.01064569, 0.0064131, 0.00378957, 0.00340306, 0.00284395, 0.00273837, 0.00270975, 0.00295056, 0.00325565,
         0.06234467, 0.06191323, 0.13741252, 0.01469963, 0.01862643, 0.01995398, 0.01688373, 0.00236162, 0.00399996,
         0.00373739, 0.00268705, 0.00272307, 0.00317967, 0.00282581, 0.0037003, 0.00242478, 0.00379247, 0.00264889,
         0.0019907, 0.00430475, 0.00344681, 0.00375757, 0.00387175, 0.00273297, 0.00351127, 0.02532812, 0.01592269,
         0.009373, 0.00743909, 0.00426626, 0.00251446, 0.00282074, 0.00240763, 0.00245225, 0.00187798, 0.00205471,
         0.00190551, 0.00257377, 0.00205906, 0.00237083, 0.00343417, 0.00286587, 0.00217424, 0.00286879, 0.00383388,
         0.0031218, 0.00414491, 0.00238943])



    # feature selection
    for name, param in model.named_parameters():
        if name == 'feature_select':
            param.requires_grad = False
            mask = torch.ones(feature_weight.size()[0])
            mask = mask.to(device)
            param.data = mask
    val_loss, r2, _ = trainer.evaluate(model, test_loader, device, loss_fn, config)
    fea_name = []
    fea_name += [name + str(i) for name in config.feature_dic.keys() for i in range(0, len(config.feature_dic[name]))]
    # print(fea_name)
    # ranking = defaultdict(list)
    # for i in range(len(fea_name)):
    #     ranking[fea_name[i]] = feature_importances_[i]

    # Create a DataFrame using a Dictionary
    data = {'feature_names': fea_name, 'feature_importance': (feature_weight).cpu().numpy()}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.clf()
    plt.figure(figsize=(20, 24))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title('VDPSelection' + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

    score = []
    ran = range(1, len(feature_weight), 3)
    print(torch.topk(feature_weight, 25, largest=True).indices)
    # print(torch.topk(feature_weight, 100, largest=True).indices)
    # print(torch.topk(feature_weight, 100, largest=True).values)

    for i in ran:
        for name, param in model.named_parameters():
            if name == 'feature_select':
                param.requires_grad = False
                mask = torch.full((1, 1, feature_weight.size()[0]), 0.0)
                mask[0, 0][torch.topk(feature_weight, i, largest=True).indices] = 1.0
                mask = mask.to(device)
                param.data = mask

        val_loss, r2, _ = trainer.evaluate(model, test_loader, device, loss_fn, config)
        #
        score.append(r2)

    print(score)
    fig = plt.figure()
    plt.plot(ran,
             score, 'o-', color='g')
    fig.suptitle("F1")
    plt.show()