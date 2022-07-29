import datetime
from collections import deque

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def time_to_timespan(timestamp):
    return int(datetime.datetime.timestamp(datetime.datetime.strptime(timestamp[:-3], "%Y-%m-%dD%H:%M:%S.%f")) * 1000)


def calculator_order_flows(curr_p, pre_p, curr_v, pre_v, is_bid=True):
    if curr_p == pre_p:
        return curr_v - pre_v
    elif curr_p > pre_p:
        if is_bid:
            return curr_v
        else:
            return 0
    else:
        if is_bid:
            return 0
        else:
            return curr_v


def bid_ask_imbalance(data, deep=4):
    bid_vol = []
    ask_vol = []
    for i in range(3, 3 + deep * 2, 2):
        bid_vol.append(data.iloc[:, i + 20].to_numpy())
        ask_vol.append(data.iloc[:, i + 21].to_numpy())
    bid_vol = np.sum(bid_vol, axis=0)
    ask_vol = np.sum(ask_vol, axis=0)
    return np.subtract(bid_vol, ask_vol) / np.add(bid_vol, ask_vol)


def get_TCI_and_TVI(books, trades, duration=1000):
    trade_vol = 0
    total_vol = 0
    trade_count = 0
    count = 0
    TC_IMBAL = []
    TV_IMBAL = []
    book_time = books['time'].to_numpy()
    trade_time = trades['time'].to_numpy()
    ini_time = book_time[0]
    book_time = book_time - ini_time
    trade_time = trade_time - ini_time
    begin, head, tail = 0, 0, 0
    print(len(book_time), len(trade_time))

    while book_time[begin] - trade_time[0] < duration:
        TC_IMBAL.append(0)
        TV_IMBAL.append(0)
        begin += 1
    while trade_time[head] < book_time[begin]:
        vol = trades.iat[head, 5]
        if trades.iat[head, 3]:
            trade_vol += vol
            trade_count += 1
        else:
            trade_vol -= vol
            trade_count -= 1
        count += 1
        total_vol += vol
        head += 1

    for i in range(begin, len(book_time) - 1):

        TC_IMBAL.append(trade_count / count if count > 0 else TC_IMBAL[-1])
        TV_IMBAL.append(trade_vol / total_vol if count > 0 else TV_IMBAL[-1])

        while head < len(trade_time) and book_time[i + 1] > trade_time[head]:
            vol = trades.iat[head, 5]
            if trades.iat[head, 3]:
                trade_vol += vol
                trade_count += 1
            else:
                trade_vol -= vol
                trade_count -= 1
            count += 1
            total_vol += vol
            head += 1

        while tail < len(trade_time) and book_time[i + 1] - trade_time[tail] > duration:
            vol = trades.iat[tail, 5]
            if trades.iat[tail, 3]:
                trade_vol -= vol
                trade_count -= 1
            else:
                trade_vol += vol
                trade_count += 1
            count -= 1
            total_vol -= vol
            tail += 1
    TC_IMBAL.append(trade_count / count if count > 0 else TC_IMBAL[-1])
    TV_IMBAL.append(trade_vol / total_vol if count > 0 else TV_IMBAL[-1])
    return TC_IMBAL, TV_IMBAL, begin


def get_midPrice_volatility(books, duration=1000):
    mid_price_volatillity = []
    window_price = deque()
    book_time = books['time'].to_numpy()
    squared_logarithmic_price = np.square(np.log(books['midPrice'].to_numpy()))
    ini_time = book_time[0]
    book_time = book_time - ini_time
    begin, tail = 0, 0

    while book_time[begin] - book_time[0] < duration:
        mid_price_volatillity.append(0)
        window_price.append(squared_logarithmic_price[begin])
        begin += 1

    for i in range(begin, len(book_time)):
        window_price.append(squared_logarithmic_price[i])
        while tail < len(book_time) and book_time[i] - book_time[tail] > duration:
            window_price.popleft()
            tail += 1
        mid_price_volatillity.append(
            np.sqrt(np.mean(window_price)) if len(window_price) > 0 else mid_price_volatillity[-1])

    return mid_price_volatillity, begin


def print_loss_graph(train_val_loss):
    train_val_loss = np.array(train_val_loss)
    fig = plt.figure()
    plt.plot([i for i in range(1, len(train_val_loss) + 1)],
             np.log10(train_val_loss[:, 0]), label='train')
    plt.plot([i for i in range(1, len(train_val_loss) + 1)],
             np.log10(train_val_loss[:, 1]), label='val')
    fig.suptitle('loss')
    plt.legend()
    plt.show()


