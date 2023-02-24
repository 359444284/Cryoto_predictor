import numpy as np
from .AbstractClassifyStrategy import BaseClassificationStrategy


class C_FBP(BaseClassificationStrategy):
    def __init__(self, row_data, model_out, config):
        super().__init__(row_data, model_out)
        self.quantile = config.quantile
        self.min_data_size = config.min_data_size
        self.get_trade_threshold()

    def get_trade_signal(self, idx):
        if idx < self.min_data_size:
            return 0

        if self.pred[idx] == 2 and self.prob[idx] >= self.buy_throds[idx]:
            return 2
        elif self.pred[idx] == 1 and self.prob[idx] >= self.sell_throd[idx]:
            return 1
        else:
            return 0

    def get_trade_threshold(self):
        probs = np.lib.stride_tricks.sliding_window_view(self.prob, self.min_data_size, axis=0)
        filter = np.lib.stride_tricks.sliding_window_view(self.pred, self.min_data_size, axis=0)

        positive_filter = filter == 2
        negative_filter = filter == 1

        positive_prob = np.where(positive_filter, probs, np.nan)
        negative_prob = np.where(negative_filter, probs, np.nan)

        np.nanquantile(positive_prob, self.quantile, axis=1, overwrite_input=True)
        np.nanquantile(negative_prob, self.quantile, axis=1, overwrite_input=True)

        np.nan_to_num(positive_prob, copy=False)
        np.nan_to_num(negative_prob, copy=False)

        self.buy_throds = np.pad(positive_prob, (self.min_data_size-1, 0), constant_values=(0))
        self.sell_throd = np.pad(negative_prob, (self.min_data_size-1, 0), constant_values=(0))





