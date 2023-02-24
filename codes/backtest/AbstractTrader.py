from abc import abstractmethod, ABC
import numpy as np
from .backtest_tools import tradebot
from .config_backtest import ConfigBacktest

class BasicTrader(ABC):
    def __init__(self, row_data):
        if type(row_data) is not np.ndarray:
            row_data = np.array(row_data)
        self.config = ConfigBacktest()
        self.mid_price = row_data[:, 0]
        self.bid = row_data[:, 1]
        self.ask = row_data[:, 2]
        self.trade_machine = tradebot(self.config.trade_vol, self.config.trade_fee, self.config.allow_short)
        self.number_of_data = len(self.bid)
        assert len(self.mid_price) == len(self.bid) == len(self.ask)


    @abstractmethod
    def run_strategy(self):
        pass

    def print_result(self):
        trade_history = self.trade_machine.get_his()
        backtest_data = [self.trade_machine.get_trade_time(),
                         self.trade_machine.get_winning_rate(),
                         self.trade_machine.get_underfee_rate(),
                         self.trade_machine.get_max_drawdown(),
                         trade_history[-1],
                         (self.mid_price / self.mid_price[0] - 1)[len(trade_history) - 1]]
        print('trade time:', backtest_data[0])
        print('winning rate:', backtest_data[1])
        print('meaningful winning:', backtest_data[2])
        print('max drawback:', backtest_data[3])
        print('highest value:', max(trade_history))
        print('profit:', backtest_data[4])
        # print('sharp: ', DL_machine.get_sharpe())
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.mid_price / self.mid_price[0] - 1, label='price', alpha=1, color='black')
        ax.plot((np.array(trade_history)), label='profit', alpha=0.7, color='red')
        ax.set_xlim(left=0, right=len(trade_history))
        plt.show()
