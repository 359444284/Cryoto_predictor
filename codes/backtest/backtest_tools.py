from collections import deque
import numpy as np
from matplotlib import pyplot as plt




# tradebot for market order
class tradebot:
    def __init__(self, volume=0.1, fee=0.02, short=True):
        self.volume = volume
        self.open_price = 0
        self.side = 0 # buy:1 , sell:-1
        self.fee = fee/100 # fee%
        self.history = [0.0]
        self.win = 0
        self.loss = 0
        self.under_fee = 0
        self.short = short

    def deal(self, price, side):
        earn = -(price / self.open_price - 1) * self.volume * side
        fee = (self.fee * self.volume + abs(earn) * self.fee)
        self.history.append(self.history[-1] + earn - fee)
        if earn > 0:
            self.win += 1
            if earn < fee:
                self.under_fee += 1
        else:
            self.loss += 1
        self.side = 0

    def buy_signal(self, ask, bid):
        if self.side == 0:
            self.open_price = ask
            self.side = 1
        elif self.side == -1 and self.short:
            self.deal(bid, 1)
            return
        self.history.append(self.history[-1])

    def sell_signal(self, ask, bid):
        if self.side == 0 and self.short:
            self.open_price = bid
            self.side = -1
        elif self.side == 1:
            self.deal(ask, -1)
            return
        self.history.append(self.history[-1])

    def add_his(self):
        self.history.append(self.history[-1])

    def get_his(self):
        return self.history

    def get_trade_time(self):
        return self.win + self.loss

    def get_winning_rate(self):
        return self.win/(self.loss + self.win + 1e-10)

    def get_underfee_rate(self):
        return 1 - (self.under_fee/(self.win + 1e-10))

    def get_max_drawdown(self):
        prue_value = (np.array(self.history) + 1)
        prefix_max = np.maximum.accumulate(prue_value)
        # i = np.argmax(prefix_max - self.history)  # 结束位置
        i = np.argmax((prefix_max - prue_value)/prefix_max)
        if i == 0:
            return 0
        j = np.argmax(prue_value[:i])  # 开始位置

        return (prue_value[j] - prue_value[i]) / (prue_value[j])


# TODO: this class should be used by trade machine (need refactor)
class trade_cache():
    def __init__(self, during=1000):
        self.during = during
        self.cache = deque()
        self.cache.append([0.0, 0.0, 0.0])
        self.tci_tvi = [0.0, 0.0, 0.0]
        self.prev_tci_tvi = [0.0, 0.0]

    def add(self, curr_time, volume, side):
        sign = 1 if side=='buy' else -1
        self.tci_tvi[0] += sign
        self.tci_tvi[1] += sign * volume
        self.tci_tvi[2] += volume
        self.cache.append([curr_time, sign, volume])

        while curr_time - self.cache[0][0] > self.during:
            last_time, last_sign, last_vol = self.cache.popleft()
            self.tci_tvi[0] -= last_sign
            self.tci_tvi[1] -= last_sign * last_vol
            self.tci_tvi[2] -= last_vol

    def get(self):
        if len(self.cache) == 0:
            tci = self.prev_tci_tvi[0]
        else:
            tci = self.tci_tvi[0] / len(self.cache)
            self.prev_tci_tvi[0] = tci

        if self.tci_tvi[2] == 0:
            tvi = self.prev_tci_tvi[1]
        else:
            tvi = self.tci_tvi[1] / self.tci_tvi[2]
            self.prev_tci_tvi[1] = tvi
        return tci, tvi










