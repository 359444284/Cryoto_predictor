import numpy as np
from .AbstractClassifyStrategy import BaseClassificationStrategy


class C_FBCS(BaseClassificationStrategy):
    def __init__(self, row_data, model_out, config):
        super().__init__(row_data, model_out)
        self.quantile = config.quantile
        self.min_data_size = config.min_data_size
        self.accumulator = [0, 0]  # (signal, times)
        self.accumulator_N = config.accumulator_N

    def get_trade_signal(self, idx):
        if self.pred[idx] == self.accumulator[0]:
            self.accumulator[1] += 1
        else:
            self.accumulator = [self.pred[idx], 1]

        if self.accumulator[1] >= self.accumulator_N:
            return self.accumulator[0]
        else:
            return 0








