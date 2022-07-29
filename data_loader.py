import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class LoadDataset:
    def __init__(self, config):
        self.config = config
        if config.name_dataset == 'crypto':
            self.inputs, self.targets = self.load_data(config)
            self.inputs = self.scale(self.inputs, config)
            self.data_num = np.shape(self.inputs)[0]
        else:
            self.train_set, self.test_set = self.load_data(config)

    # def scale(self, data, config):
    #     mean = np.mean(data, axis=0)
    #     std = np.std(data, axis=0)
    #     return (data - mean) / (std + 1e-10)

    def scale(self, data, config):
        num_pre_day = len(data) // config.num_of_day
        mean = np.mean(data[:num_pre_day], axis=0)
        std = np.std(data[:num_pre_day], axis=0)
        for i in range(1, config.num_of_day + 1):
            if i == 1:
                data[:num_pre_day * i] = (data[:num_pre_day * i] - mean) / (std + 1e-10)
            elif i == config.num_of_day:
                data[num_pre_day * (i - 1):] = (data[num_pre_day * (i - 1):] - mean) / (std + 1e-10)
            else:
                tmp = (data[num_pre_day * (i - 1):num_pre_day * i] - mean) / (std + 1e-10)
                mean = np.mean(data[num_pre_day * (i - 1):num_pre_day * i], axis=0)
                std = np.std(data[num_pre_day * (i - 1):num_pre_day * i], axis=0)
                data[num_pre_day * (i - 1):num_pre_day * i] = tmp

        print(type(data))
        return data

    def load_data(self, config):
        if config.name_dataset == 'crypto':
            if self.config.debug_mode:
                if self.config.use_all_features:
                    data = pd.read_csv(self.config.data_path, nrows=self.config.debug_num)
                else:
                    data = pd.read_csv(self.config.data_path, nrows=self.config.debug_num,
                                       usecols=self.config.feature_index)
            else:
                if self.config.use_all_features:
                    data = pd.read_csv(self.config.data_path)
                else:
                    data = pd.read_csv(self.config.data_path, usecols=self.config.feature_index)

            return data.iloc[:, 2:].to_numpy(), data['mid_price'].to_numpy()
        else:
            usecols = [i for i in range(40)] + [i for i in range(145, 150)]
            if self.config.debug_mode:
                train_data = pd.read_csv(self.config.data_path + 'train.csv', nrows=self.config.debug_num,
                                         usecols=usecols)
                test_data = pd.read_csv(self.config.data_path + 'test.csv', nrows=self.config.debug_num,
                                        usecols=usecols)
            else:
                train_data = pd.read_csv(self.config.data_path + 'train.csv',
                                         usecols=usecols, header=None)
                test_data = pd.read_csv(self.config.data_path + 'test.csv',
                                        usecols=usecols, header=None)
            return train_data.to_numpy(), test_data.to_numpy()

    # adapted from: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/blob/master/jupyter_pytorch/run_train_pytorch.ipynb
    def get_FI_data(self, dataset='train'):
        if dataset == 'train':
            train_num = int(0.8 * len(self.train_set))
            x = self.train_set[:train_num, :40]
            y = self.train_set[:train_num, -5:]
        elif dataset == 'val':
            train_num = int(0.8 * len(self.train_set))
            x = self.train_set[train_num:, :40]
            y = self.train_set[train_num:, -5:]
        else:
            x = self.test_set[:, :40]
            y = self.test_set[:, -5:]

        [N, D] = x.shape
        T = self.config.lockback_window
        dataY = y[T - 1:N]
        dataX = np.zeros((N - T + 1, T, D))
        for i in range(T, N + 1):
            dataX[i - T] = x[i - T:i, :]

        dataY = dataY[:, self.config.k] - 1
        return dataX, dataY

    def get_crypto_data(self, datatype='train'):
        assert (datatype in ['train', 'val', 'test'])
        train_val_num = int(self.config.train_val_ratio * self.data_num)
        train_num = int(self.config.train_ratio * train_val_num)
        if datatype == 'train':
            train_num = int(self.config.train_ratio * train_val_num)
            features = self.inputs[:train_num]
            log_mid_price = np.log(self.targets[:train_num])
        elif datatype == 'val':
            features = self.inputs[train_num:train_val_num]
            log_mid_price = np.log(self.targets[train_num:train_val_num])
        else:
            features = self.inputs[train_val_num:]
            log_mid_price = np.log(self.targets[train_val_num:])

        X, Y = features[:-self.config.forecast_horizon], \
               log_mid_price[self.config.lockback_window - 1:]

        assert abs(len(X) - len(Y)) == abs((self.config.lockback_window - 1) - self.config.forecast_horizon)
        if self.config.preprocess:
            Y = np.concatenate([[Y[:-self.config.forecast_horizon] -
                                 (Y[i:(-self.config.forecast_horizon + i)] if i < self.config.forecast_horizon else Y[
                                                                                                                    i:])]
                                for i in range(1, self.config.forecast_horizon + 1, self.config.forecast_stride)],
                               axis=0).T
            X = np.lib.stride_tricks.sliding_window_view(X, self.config.lockback_window, axis=0).transpose(0, 2, 1)
            assert (len(X) == len(Y))

        return X, Y


class ProcessDataset(Dataset):
    def __init__(self, features, targets, with_label, config, regression=True):
        self.config = config
        self.inputs = features
        self.regression = regression
        if with_label:
            self.targets = targets
        self.with_label = with_label

    def __len__(self):
        if not self.config.preprocess:
            return len(self.inputs) - max(self.config.lockback_window, self.config.forecast_horizon)
        else:
            return len(self.inputs)

    def __getitem__(self, item):
        if not self.config.preprocess:
            feature = self.inputs[item:(item + self.config.lockback_window)]
            target = np.array([self.targets[item + j] - self.targets[item]
                               for j in range(1, self.config.forecast_horizon + 1, self.config.forecast_stride)])
            if self.targets[item + self.config.forecast_horizon] == 0:
                raise ValueError
        else:
            feature = self.inputs[item]
            target = self.targets[item]

        if self.with_label:
            return {
                'input': torch.tensor(feature, dtype=torch.float32),
                'target': torch.tensor(target, dtype=(torch.float32 if self.regression else torch.long))
            }
        else:
            return {
                'input': torch.tensor(feature, dtype=torch.float32)
            }
