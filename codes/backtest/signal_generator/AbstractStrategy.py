from abc import abstractmethod, ABC
import numpy as np


class BaseStrategy(ABC):
    """
    :param row_data: a list contain [mid_price, bid1, ask1]
    :param pred: trade signal from prediction pipeline: 0 hold, 1 sell, 2 buy
    """
    def __init__(self, row_data):
        self.name = str(type(self))
        self.bid = row_data[:, 1]
        self.ask = row_data[:, 2]

    @abstractmethod
    def get_trade_signal(self, idx):
        pass
