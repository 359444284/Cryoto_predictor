import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.timefeatures import time_features


def scale_function(data, mean, std):
    return ((data - mean) / (std + 1e-8)).astype('f')


class LoadDataset:
    def __init__(self, config):
        self.config = config
        if config.name_dataset == 'fi2010':
            self.train_set, self.test_set = self.load_data()
        else:
            self.date_time, self.inputs, self.targets = self.load_data()
            # self.inputs = self.scale(self.inputs, train=True)


    def scale(self, data, train=True):
        datalist = data
        if train:
            mean = np.mean(datalist[0], axis=0)[3:]
            std = np.std(datalist[0], axis=0)[3:]
            for i in range(len(datalist)):
                if i == 0:
                    datalist[0][:, 3:] = scale_function(datalist[0][:, 3:], mean, std)
                else:
                    tmp = scale_function(datalist[i][:, 3:], mean, std)
                    mean = np.mean(datalist[i], axis=0)[3:]
                    std = np.std(datalist[i], axis=0)[3:]
                    datalist[i][:, 3:] = tmp
        else:
            mean = np.mean(datalist[0], axis=0)
            std = np.std(datalist[0], axis=0)
            for i in range(len(datalist)):
                if i == 0:
                    datalist[0] = scale_function(datalist[0], mean, std)
                else:
                    tmp = scale_function(datalist[i], mean, std)
                    mean = np.mean(datalist[i], axis=0)
                    std = np.std(datalist[i], axis=0)
                    datalist[i] = tmp

        return datalist

    def load_csv(self, path, usecol=None):
        if usecol:
            pass
        else:
            usecol = self.config.feature_index
        if self.config.use_all_features:
            if self.config.debug_mode:
                data = pd.read_csv(path, nrows=self.config.debug_num, usecols=usecol)
            else:
                data = pd.read_csv(path, usecols=usecol)
        else:
            if self.config.debug_mode:
                data = pd.read_csv(path, nrows=self.config.debug_num, usecols=usecol)
            else:
                data = pd.read_csv(path, usecols=usecol)
        return data

    def load_data(self):
        if self.config.name_dataset == 'fi2010':
            usecols = [i for i in range(40)] + [i for i in range(145, 150)]
            if self.config.debug_mode:
                train_data = pd.read_csv(self.config.data_path + 'train.csv', nrows=self.config.debug_num,
                                         usecols=usecols, header=None)
                test_data = pd.read_csv(self.config.data_path + 'test.csv', nrows=self.config.debug_num,
                                        usecols=usecols, header=None)
            else:
                train_data = pd.read_csv(self.config.data_path + 'train.csv',
                                         usecols=usecols, header=None)
                test_data = pd.read_csv(self.config.data_path + 'test.csv',
                                        usecols=usecols, header=None)
            return train_data.to_numpy(), test_data.to_numpy()
        else:
            input_list = []
            target_list = []
            date_list = []
            for i in range(self.config.split_data[0], self.config.split_data[-1]):
                file_path = self.config.data_path + self.config.file_name[i]
                data = self.load_csv(file_path)
                if i == 0:
                    print(data.head(1))
                if self.config.use_time_feature:
                    date_list.append(data['time'])
                if self.config.regression:
                    input_list.append(data.iloc[:, 1:].to_numpy())
                    target_list.append(data['mid_price'].to_numpy())
                else:
                    input_list.append(data.iloc[:, 1:-1].to_numpy())
                    target_list.append(data.iloc[:, -1].to_numpy())
            return date_list, input_list, target_list

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
        return {
            'X': dataX,
            'Y': dataY
        }

    def get_crypto_data(self, datatype='train'):
        assert (datatype in ['train', 'val', 'test'])

        start_train, end_train, end_test = self.config.split_data
        start_train, end_train, end_test = 0, end_train-start_train, end_test-start_train
        if datatype in ['train', 'val']:
            if self.config.use_time_feature:
                train_val_date = pd.concat(self.date_time[:end_train], ignore_index=True)
            train_val_input = np.concatenate(self.inputs[:end_train])
            train_val_target = np.concatenate(self.targets[:end_train])
            train_num = int(self.config.train_ratio * -len(train_val_input))
            if datatype == 'train':
                if self.config.use_time_feature:
                    date = train_val_date[train_num:]
                features = train_val_input[train_num:]
                target = train_val_target[train_num:]
            else:
                if self.config.use_time_feature:
                    date = train_val_date[:train_num]
                features = train_val_input[:train_num]
                target = train_val_target[:train_num]
        else:
            if self.config.use_time_feature:
                date = pd.concat(self.date_time[end_train:], ignore_index=True)
            test_input = np.concatenate(self.inputs[end_train:])
            test_target = np.concatenate(self.targets[end_train:])
            features = test_input
            target = test_target

        if self.config.regression:
            target = np.log(target)
            # pass
        else:
            pass
        if self.config.use_time_feature:
            data_stamp = time_features(pd.to_datetime(date.values, unit='ms'), freq=self.config.freq)
            data_stamp = data_stamp.transpose(1, 0)
            X, Y = features, target

            if self.config.regression:
                data_stamp = data_stamp[self.config.forecast_horizon:]
                X = features[self.config.forecast_horizon:]
                # Y = target[self.config.forecast_horizon:]
                Y = pd.DataFrame(target)
                Y = (Y - Y.shift(self.config.forecast_horizon)).to_numpy()[self.config.forecast_horizon:]
            else:
                X = np.lib.stride_tricks.sliding_window_view(X, self.config.lockback_window, axis=0).transpose(0, 2, 1)
                Y = Y[self.config.lockback_window - 1:]
                data_stamp = np.lib.stride_tricks.sliding_window_view(data_stamp, self.config.lockback_window, axis=0).transpose(0, 2, 1)
                assert (len(X) == len(Y) == len(data_stamp))

            # log_mp = np.expand_dims(log_mid_price[:-self.config.forecast_horizon], axis=-1)
            # Y = pd.DataFrame(log_mid_price)
            # Y = np.squeeze(Y.rolling(30).mean().to_numpy()[30:])
            # Y = (Y - Y.shift(self.config.forecast_horizon)).to_numpy()
            # data = np.concatenate([features[self.config.forecast_horizon:], Y[self.config.forecast_horizon:]], axis=1)
            # target = Y[self.config.forecast_horizon:]
            # data = features[:-self.config.forecast_horizon]
            return {
                'stamp': data_stamp,
                'X': X,
                'Y': Y
            }
        else:
            X, Y = features, target

            if self.config.preprocess:
                print('pre_processing')
                if self.config.regression:
                    Y = np.concatenate([[Y[:-self.config.forecast_horizon] -
                                         (Y[i:(-self.config.forecast_horizon + i)] if i < self.config.forecast_horizon else Y[
                                                                                                                            i:])]
                                        for i in range(1, self.config.forecast_horizon + 1, self.config.forecast_stride)],
                                       axis=0).T
                    X = np.lib.stride_tricks.sliding_window_view(X, self.config.lockback_window, axis=0).transpose(0, 2, 1)
                else:
                    X = np.lib.stride_tricks.sliding_window_view(X, self.config.lockback_window, axis=0).transpose(0, 2, 1)
                    # mean = np.mean(X, axis=1)
                    # print(mean.shape)
                    # std_list = []
                    # chunk = 20
                    # num_chunk = len(X)//chunk
                    # for i in range(chunk):
                    #     if i != chunk-1:
                    #         std = np.std(X[num_chunk*i:num_chunk*(i+1)], axis=1)
                    #     else:
                    #         std = np.std(X[num_chunk * i:], axis=1)
                    #     std_list.append(std)
                    # print('ok')
                    # std = np.concatenate(std_list, axis=0)
                    # std = std + 1e-8
                    # print(std.shape)
                    # X = X[:, -1, :]
                    #
                    # X = np.subtract(X, mean)
                    # X /= std
                    #
                    #
                    # print('ok2')

                    Y = Y[self.config.lockback_window-1:]

                print(Y.shape)
                print(X.shape)
                assert (len(X) == len(Y))
            return {
                'X': X,
                'Y': Y
            }


class ProcessDataset(Dataset):
    def __init__(self, data_dic, with_label, config):
        self.config = config
        self.inputs = data_dic['X']
        self.regression = self.config.regression
        self.targets = data_dic['Y']
        if config.use_time_feature:
            self.stamp = data_dic['stamp']
        self.with_label = with_label

    def __len__(self):
        if not self.config.preprocess or self.regression:
            # return len(self.inputs) - max(self.config.lockback_window, self.config.forecast_horizon)
            return len(self.inputs) - self.config.lockback_window - self.config.forecast_horizon + 1
        else:
            return len(self.inputs)

    def __getitem__(self, item):
        if self.config.use_time_feature:
            if not self.regression:
                feature = self.inputs[item].copy()
                target = self.targets[item]
                feature_mask = self.stamp[item]

                mean = np.mean(feature, axis=0, keepdims=True)[:, 3:]
                std = np.std(feature, axis=0, keepdims=True)[:, 3:]
                feature[:, 3:] = (feature[:, 3:] - mean) / (std + 1e-8)

                return {
                    'input': torch.tensor(feature, dtype=torch.float32),
                    'target': torch.tensor(target, dtype=(torch.float32 if self.regression else torch.long)),
                    'inp_mask': torch.tensor(feature_mask, dtype=torch.float32),
                }
            else:
                s_begin = item
                s_end = s_begin + self.config.lockback_window
                r_begin = s_end - self.config.lockback_window//2
                pre_star = s_end
                r_end = pre_star + self.config.forecast_horizon
                seq_x = self.inputs[s_begin:s_end]
                seq_x = np.concatenate([seq_x, self.targets[s_begin:s_end].reshape(-1, 1)], axis=1)

                # seq_x = np.concatenate([seq_x, np.array([self.targets[s_end-1 - j] - self.targets[s_end-1]
                #                                          for j in range(1, self.config.lockback_window + 1,
                #                                                         1)]).reshape(-1, 1)], axis=1)

                # seq_y = np.array([self.targets[pre_star - 1 + j] - self.targets[pre_star-1]
                #                        for j in range(1, self.config.forecast_horizon + 1, self.config.forecast_stride)])
                # seq_y = self.targets[pre_star:r_end]
                seq_y = self.targets[pre_star:r_end].reshape(-1, 1)

                # source = self.targets[pre_star-1:r_end]
                seq_y = np.concatenate([self.inputs[pre_star:r_end], seq_y], axis=1)
                # seq_y = self.inputs[r_begin:r_end]
                # seq_y = np.concatenate([self.inputs[r_begin:r_end] - self.inputs[pre_star-1], seq_y.reshape(-1, 1)], axis=1)
                seq_x_mark = self.stamp[s_begin:s_end]
                seq_y_mark = self.stamp[r_begin:r_end]
                return {
                    'input': torch.tensor(seq_x, dtype=torch.float32),
                    'target': torch.tensor(seq_y, dtype=(torch.float32 if self.regression else torch.long)),
                    'inp_mask': torch.tensor(seq_x_mark, dtype=torch.float32),
                    'tar_mask': torch.tensor(seq_y_mark, dtype=(torch.float32)),
                }
        else:
            if not self.config.preprocess:
                feature = self.inputs[item:(item + self.config.lockback_window)]
                target = np.array([self.targets[item + j] - self.targets[item]
                                   for j in range(1, self.config.forecast_horizon + 1, self.config.forecast_stride)])
                if self.targets[item + self.config.forecast_horizon] == 0:
                    raise ValueError
            else:
                feature = self.inputs[item].copy()
                target = self.targets[item]
                mean = np.mean(feature, axis=0, keepdims=True)[:, 3:]
                std = np.std(feature, axis=0, keepdims=True)[:, 3:]
                feature[:, 3:] = (feature[:, 3:] - mean) / (std + 1e-8)
                # mean = np.mean(feature, axis=0, keepdims=True)
                # std = np.std(feature, axis=0, keepdims=True)
                # feature = (feature - mean) / (std + 1e-8)

            return {
                    'input': torch.tensor(feature, dtype=torch.float32),
                    'target': torch.tensor(target, dtype=(torch.float32 if self.regression else torch.long))
                }



