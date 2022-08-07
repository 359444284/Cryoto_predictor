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

    model.load(last=False)

    model = model.to(device)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    optimizer = Adam(model.parameters(), lr=config.lr)

    if config.regression:
        loss_fn = nn.MSELoss(reduction='mean').to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    val_loss, r2 = trainer.evaluate(model, test_loader, device, loss_fn, config)
    # #
    print(r2)
    if config.name_dataset == 'crypto':
        fig = plt.figure()
        plt.plot([i for i in range(1, config.forecast_horizon + 1, 1)],
                 np.clip(r2, -0.2, 1), 'o-', color='g')
        fig.suptitle("test set: " + str(max(r2)))
        plt.show()
    else:
        print(r2)

#     model = getattr(model_card, 'LSTM')()
#     model.load()
#     if config.select_fun:
#         model = getattr(model_card, config.select_fun_dic[config.select_fun])(model)
#     # model.load(last=False)
#     #
#     # model = model.model
#     # if config.select_fun:
#     #     model = getattr(model_card, 'FS')(model)
#     model = model.to(device)
#     for name, param in model.named_parameters():
#         print(name)
#     optimizer = Adam(model.parameters(), lr=config.lr)
#     if config.regression:
#         loss_fn = nn.MSELoss(reduction='mean').to(device)
#     else:
#         loss_fn = nn.CrossEntropyLoss().to(device)
#     # feature_weight = torch.sigmoid(torch.load('./checkpoints/last_lstm_FS_allfea_100_3_.pt')['p_logit'].detach())
#     # feature_weight = torch.sigmoid(torch.load('./checkpoints/best_lstm_FS_allfea_100_3_.pt')['feature_select'].detach())
#     feature_weight = torch.tensor([0.00307308, 0.00219175, 0.00222428, 0.00206057, 0.00213096, 0.00235912, 0.00175697, 0.00183962, 0.00243541, 0.00218106, 0.00220243, 0.00215446, 0.0018324, 0.00192439, 0.00260901, 0.00229553, 0.00215351, 0.00247028, 0.01765554, 0.01423167, 0.00195414, 0.0019417, 0.00190619, 0.00203582, 0.00171267, 0.00166206, 0.00194251, 0.00243371, 0.00169414, 0.00197938, 0.00171387, 0.00166885, 0.00178335, 0.0018351, 0.00190884, 0.0014969, 0.00198518, 0.00152204, 0.05044684, 0.05609723, 0.00769489, 0.00283361, 0.00390799, 0.00278641, 0.00238566, 0.00238346, 0.00246087, 0.00203306, 0.00262252, 0.00288963, 0.0029829, 0.00317745, 0.00545727, 0.00206057, 0.00311994, 0.00399826, 0.00603879, 0.00697224, 0.01716152, 0.01105127, 0.00310578, 0.00285695, 0.00300768, 0.00387422, 0.00358696, 0.00283522, 0.00299954, 0.00302657, 0.00355144, 0.00295965, 0.00365416, 0.00332215, 0.00312583, 0.00258763, 0.00321038, 0.00242006, 0.00344352, 0.00302744, 0.00528374, 0.00694191, 0.00572423, 0.00314403, 0.00276517, 0.002984, 0.0024025, 0.00204968, 0.00260858, 0.0014968, 0.01124795, 0.01193252, 0.00589066, 0.00525919, 0.00299138, 0.00341358, 0.00218256, 0.00345053, 0.0025149, 0.00286558, 0.01225451, 0.01064569, 0.0064131, 0.00378957, 0.00340306, 0.00284395, 0.00273837, 0.00270975, 0.00295056, 0.00325565, 0.06234467, 0.06191323, 0.13741252, 0.01469963, 0.01862643, 0.01995398, 0.01688373, 0.00236162, 0.00399996, 0.00373739, 0.00268705, 0.00272307, 0.00317967, 0.00282581, 0.0037003, 0.00242478, 0.00379247, 0.00264889, 0.0019907, 0.00430475, 0.00344681, 0.00375757, 0.00387175, 0.00273297, 0.00351127, 0.02532812, 0.01592269, 0.009373, 0.00743909, 0.00426626, 0.00251446, 0.00282074, 0.00240763, 0.00245225, 0.00187798, 0.00205471, 0.00190551, 0.00257377, 0.00205906, 0.00237083, 0.00343417, 0.00286587, 0.00217424, 0.00286879, 0.00383388, 0.0031218, 0.00414491, 0.00238943]
# )
#
#     # feature selection
#     for name, param in model.named_parameters():
#         if name == 'feature_select':
#             param.requires_grad = False
#             mask = torch.ones(feature_weight.size()[0])
#             mask = mask.to(device)
#             param.data = mask
#     val_loss, r2 = trainer.evaluate(model, test_loader, device, loss_fn, config)
#     fea_name = []
#     fea_name += [name + str(i) for name in config.feature_dic.keys() for i in range(0, len(config.feature_dic[name]))]
#     # print(fea_name)
#     # ranking = defaultdict(list)
#     # for i in range(len(fea_name)):
#     #     ranking[fea_name[i]] = feature_importances_[i]
#
#     # Create a DataFrame using a Dictionary
#     data = {'feature_names': fea_name, 'feature_importance': (feature_weight).cpu().numpy()}
#     fi_df = pd.DataFrame(data)
#
#     # Sort the DataFrame in order decreasing feature importance
#     fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
#
#     # Define size of bar plot
#     plt.clf()
#     plt.figure(figsize=(20, 24))
#     # Plot Searborn bar chart
#     sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#     # Add chart labels
#     plt.title('VDPSelection' + 'FEATURE IMPORTANCE')
#     plt.xlabel('FEATURE IMPORTANCE')
#     plt.ylabel('FEATURE NAMES')
#     plt.show()
#
#     score = []
#     ran = range(5, 105, 5)
#     print(torch.topk(feature_weight, 51, largest=True).indices)
#     print(torch.topk(feature_weight, 100, largest=True).indices)
#     print(torch.topk(feature_weight, 100, largest=True).values)

    # for i in ran:
    #     for name, param in model.named_parameters():
    #         if name == 'feature_select':
    #             param.requires_grad = False
    #             mask = torch.full((1, 1, feature_weight.size()[0]), 0.0)
    #             mask[0, 0][torch.topk(feature_weight, i, largest=False).indices] = 1.0
    #             mask = mask.to(device)
    #             param.data = mask
    #
    #     val_loss, r2 = trainer.evaluate(model, test_loader, device, loss_fn, config)
    #     #
    #     score.append(r2)
    #
    # print(score)
    # fig = plt.figure()
    # plt.plot(ran,
    #          score, 'o-', color='g')
    # fig.suptitle("F1")
    # plt.show()
