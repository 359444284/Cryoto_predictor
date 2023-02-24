import collections

from codes.backtest import signal_generator
from .AbstractTrader import BasicTrader


class MarketTrader(BasicTrader):
    def __init__(self, row_data, model_out, strategy_name="C_FBP"):
        super().__init__(row_data)
        self.strategy = getattr(signal_generator, strategy_name)(row_data, model_out, self.config)

    def run_strategy(self):
        for i in range(self.number_of_data - self.config.trade_delay):
            trade_signal = self.strategy.get_trade_signal(i)
            if trade_signal == 2:
                self.trade_machine.buy_signal(self.ask[i + self.config.trade_delay], self.bid[i + self.config.trade_delay])
            elif trade_signal == 1:
                self.trade_machine.sell_signal(self.ask[i + self.config.trade_delay], self.bid[i + self.config.trade_delay])
            else:
                self.trade_machine.add_his()
