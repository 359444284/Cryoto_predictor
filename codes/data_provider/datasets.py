import numpy as np
import torch
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(self, data_dic, with_label, config):
        self.config = config
        self.inputs = data_dic['X']
        self.regression = self.config.regression
        self.targets = data_dic['Y']
        if config.use_time_feature:
            self.stamp = data_dic['stamp']
        self.with_label = with_label

"""
This dataset load preprocessed data
"""
class ClassicDataset(AbstractDataset):
    def __init__(self, data_dic, with_label, config):
        super().__init__(data_dic, with_label, config)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        feature = self.inputs[item].copy()
        target = self.targets[item]
        if self.config.Normalizer == 'LC-Norm':
            mean = np.mean(feature, axis=0, keepdims=True)[:, 3:]
            std = np.std(feature, axis=0, keepdims=True)[:, 3:]
            feature[:, 3:] = (feature[:, 3:] - mean) / (std + 1e-8)
            feature = feature[-self.config.lockback_window:, :]

        if self.config.use_time_feature:
            feature_mask = self.stamp[item]
            return {
                'input': torch.tensor(feature, dtype=torch.float32),
                'target': torch.tensor(target, dtype=(torch.float32 if self.regression else torch.long)),
                'inp_mask': torch.tensor(feature_mask, dtype=torch.float32),
            }
        else:
            return {
                'input': torch.tensor(feature, dtype=torch.float32),
                'target': torch.tensor(target, dtype=(torch.float32 if self.regression else torch.long))
            }


"""
This dataset process data in side the get() function
"""
class ProcessDataSet(AbstractDataset):
    def __init__(self, data_dic, with_label, config):
        super().__init__(data_dic, with_label, config)

    def __len__(self):
        return len(self.inputs) - self.config.lockback_window - self.config.forecast_horizon + 1

    def __getitem__(self, item):
        if self.config.use_time_feature:
            s_begin = item
            s_end = s_begin + self.config.lockback_window
            r_begin = s_end - self.config.lockback_window//2
            r_end = s_end + self.config.forecast_horizon

            seq_x = self.inputs[s_begin:s_end]
            # seq_x = np.concatenate([seq_x, self.targets[s_begin:s_end].reshape(-1, 1)], axis=1)
            if self.config.Normalizer == 'LC-Norm':
                mean = np.mean(seq_x, axis=0, keepdims=True)[:, 3:]
                std = np.std(seq_x, axis=0, keepdims=True)[:, 3:]
                seq_x[:, 3:] = (seq_x[:, 3:] - mean) / (std + 1e-8)
                seq_x = seq_x[-self.config.lockback_window:, :]

            # seq_y = self.targets[pre_star:r_end]
            seq_y = np.log(self.targets[r_end-1] / self.targets[s_end-1])
            seq_x_mark = self.stamp[s_begin:s_end]
            seq_y_mark = self.stamp[r_begin:r_end]
            return {
                'input': torch.tensor(seq_x, dtype=torch.float32),
                'target': torch.tensor(seq_y, dtype=(torch.float32 if self.regression else torch.long)),
                'inp_mask': torch.tensor(seq_x_mark, dtype=torch.float32),
                'tar_mask': torch.tensor(seq_y_mark, dtype=(torch.float32)),
            }
        else:
            s_begin = item
            s_end = s_begin + self.config.lockback_window
            r_end = s_end + self.config.forecast_horizon

            seq_x = self.inputs[s_begin:s_end]
            if self.config.Normalizer == 'LC-Norm':
                mean = np.mean(seq_x, axis=0, keepdims=True)[:, 3:]
                std = np.std(seq_x, axis=0, keepdims=True)[:, 3:]
                seq_x[:, 3:] = (seq_x[:, 3:] - mean) / (std + 1e-8)
                seq_x = seq_x[-self.config.lockback_window:, :]

            seq_y = [np.log(self.targets[i] / self.targets[s_end - 1]) for i in range(s_end, r_end, self.config.forecast_stride)]


            return {
                'input': torch.tensor(seq_x, dtype=torch.float32),
                'target': torch.tensor(seq_y, dtype=(torch.float32 if self.regression else torch.long))
            }


def get_dataset(data, with_label, config):
    if config.preprocess:
        print('ClassicDataset')
        return ClassicDataset(data, with_label, config)
    else:
        print('ProcessDataSet')
        return ProcessDataSet(data, with_label, config)
