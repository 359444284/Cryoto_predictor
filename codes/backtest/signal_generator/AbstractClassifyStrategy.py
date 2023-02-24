from abc import abstractmethod
from .AbstractStrategy import BaseStrategy
import numpy as np


class BaseClassificationStrategy(BaseStrategy):
    """
    :param row_data: a list contain [mid_price, bid1, ask1]
    :param model_out: model output dict [pred: predict signal, prob: confidential for signal]
    """
    def __init__(self, row_data, model_out):
        super().__init__(row_data)
        self.name = str(type(self))
        self.pred = np.array(model_out["pred"])
        self.prob = np.array(model_out["prob"])
        assert len(self.bid) == len(self.ask) == len(self.pred) == len(self.prob)

    @abstractmethod
    def get_trade_signal(self, idx):
        pass
