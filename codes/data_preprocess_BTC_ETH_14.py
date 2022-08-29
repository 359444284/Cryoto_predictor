import pandas as pd
import datetime
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import os
import datetime
from collections import deque
import tools

# books_dir = 'G:\\我的云端硬盘\\crypto_predict\\books\\'
# trades_dir = 'G:\\我的云端硬盘\\crypto_predict\\trades\\'
books_dir = './data/ETH_14_data/book/'
trades_dir = './data/ETH_14_data/trade/'
target_dir = './data/ETH_14/'

book_files = sorted(os.listdir(books_dir))
print(len(book_files))
print(book_files)
trade_files = sorted(os.listdir(trades_dir))
print(trade_files)
# books = pd.concat([pd.read_csv(books_dir + book_files[i]) for i in range(7)])
# print('before drop: ', len(books))
# books.drop_duplicates(inplace=True)
# print('after drop: ', len(books))
begin_day = 0
print(books_dir + book_files[begin_day])
print(books_dir + book_files[len(book_files)-1])
for dy in range(begin_day, len(book_files)):
    print(books_dir + book_files[dy])
    books = pd.read_csv(books_dir + book_files[dy])
    trades = pd.read_csv(trades_dir + trade_files[dy])
    if dy < len(book_files)-1:
        books = pd.concat([books, pd.read_csv(books_dir + book_files[dy+1], nrows=500)], ignore_index=True)
        trades = pd.concat([trades, pd.read_csv(trades_dir + trade_files[dy+1], nrows=500)], ignore_index=True)
    else:
        pass
    # print(books.isnull().sum())
    print('before drop: ', len(books))
    books.drop_duplicates(subset=['time'], keep='last', inplace=True)
    print('after drop: ', len(books))

    # print(trades.isnull().sum())
    print('before drop: ', len(trades))
    trades.drop_duplicates(inplace=True)
    print('after drop: ', len(trades))

    # for columnname in books.columns:
    #     if books[columnname].dtype == 'float64':
    #         books[columnname] = books[columnname].astype('float32')
    # for columnname in trades.columns:
    #     if trades[columnname].dtype == 'float64':
    #         trades[columnname] = trades[columnname].astype('float32')

    trades['side'] = trades['side'].apply(lambda x: True if x == 1 else False)
    books.info()
    trades.info()
    print(books[:1])
    print(trades[:1])

    books['time'] = books['time'].apply(tools.time_to_timespan)
    trades['time'] = trades['time'].apply(tools.time_to_timespan)

    gap = books['time'].to_numpy()
    gap = [gap[i + 1] - gap[i] for i in range(len(gap) - 1)]
    print('max gap: ', max(gap), '  min gap: ', min(gap))
    # max_pos = np.argmax(gap)
    # min_pos = np.argmin(gap)
    # print(books[(max_pos - 3):(max_pos + 3)])
    # print(books[(min_pos - 3):(min_pos + 3)])
    # print(pd.DataFrame(gap).describe().round(3))

    gap = trades['time'].to_numpy()
    gap = [gap[i + 1] - gap[i] for i in range(len(gap) - 1)]
    print('max gap: ', max(gap), '  min gap: ', min(gap))
    # max_pos = np.argmax(gap)
    # min_pos = np.argmin(gap)
    # print(trades[(max_pos - 3):(max_pos + 3)])
    # print(trades[(min_pos - 3):(min_pos + 3)])
    # print(pd.DataFrame(gap).describe().round(3))

    del gap

    books['midPrice'] = ((books['ask1'] + books['bid1']) / 2)
    books['midPrice'].describe().round(3)



    # df = books[['min', 'midPrice']].groupby('min').agg('mean')
    # print(df.head)
    # df.plot()
    # plt.xlabel('min')
    # plt.ylabel('midPrice')
    # plt.show()

    if dy > begin_day:
        curr_idx = len(prev_book)
        books = pd.concat([prev_book, books], ignore_index=True)
        trades = pd.concat([prev_trade, trades], ignore_index=True)

    best_bid = books.iloc[:, 3].to_numpy().reshape(-1, 1)
    best_ask = books.iloc[:, 13].to_numpy().reshape(-1, 1)
    best_bid_vol = books.iloc[:, 23].to_numpy().reshape(-1, 1)
    best_ask_vol = books.iloc[:, 33].to_numpy().reshape(-1, 1)
    mid_price = books['midPrice'].to_numpy().reshape(-1, 1)

    bof = []
    aof = []
    for i in range(3, 13, 1):
        bid_price = books.iloc[:, i].to_numpy()
        ask_price = books.iloc[:, i + 10].to_numpy()
        bid_vol = books.iloc[:, i + 20].to_numpy()
        ask_vol = books.iloc[:, i + 30].to_numpy()
        tmp_bof = []
        tmp_aof = []
        for i in range(1, len(bid_price)):
            tmp_bof.append(tools.calculator_order_flows(bid_price[i], bid_price[i - 1], bid_vol[i], bid_vol[i - 1]))
            tmp_aof.append(
                tools.calculator_order_flows(ask_price[i], ask_price[i - 1], ask_vol[i], ask_vol[i - 1], is_bid=False))
        bof.append(tmp_bof)
        aof.append(tmp_aof)
    # #
    ofi_features = np.concatenate([np.subtract(bof[i], aof[i]).reshape(len(bof[i]), 1) for i in range(len(bof))], axis=1)
    # #
    of_features = np.concatenate([np.array(bof).T, np.array(aof).T], axis=1)

    bid_ask_imbs = np.concatenate([[tools.bid_ask_imbalance(books, deep=i, version1=False)] for i in range(1, 11, 1)], axis=0).T

    w_volume_bid = np.sum(books.iloc[:, 23:33].to_numpy() * books.iloc[:, 3:13].to_numpy() * np.arange(1, 0, -0.1), axis=1)
    w_volume_ask = np.sum(books.iloc[:, 33:43].to_numpy() * books.iloc[:, 13:23].to_numpy() * np.arange(1, 0, -0.1), axis=1)
    w_volume_price_spread = np.subtract(w_volume_bid, w_volume_ask) / np.add(w_volume_bid, w_volume_ask)

    bid_diff = np.subtract(books.iloc[:, 4:13].to_numpy(), books.iloc[:, 3:12].to_numpy())
    ask_diff = np.subtract(books.iloc[:, 14:23].to_numpy(), books.iloc[:, 13:22].to_numpy())
    bid_add_ask = np.add(bid_diff, ask_diff)
    bid_sub_ask = np.subtract(bid_diff, ask_diff)
    height_imb = np.divide(bid_sub_ask, bid_add_ask, out=np.zeros_like(bid_sub_ask), where=bid_add_ask != 0)

    ask_press = np.sum(books.iloc[:, 33:43].to_numpy() * (mid_price / (books.iloc[:, 13:23].to_numpy() - mid_price)), axis=1)
    bid_press = np.sum(books.iloc[:, 23:33].to_numpy() * (mid_price / (mid_price - books.iloc[:, 3:13].to_numpy())), axis=1)
    price_press = np.log(ask_press) - np.log(bid_press)


    mean_vol = np.sum(books.iloc[:,23:43].to_numpy(), axis=1).reshape(-1, 1)
    S_P = np.concatenate([-np.subtract(books.iloc[:, 4:13].to_numpy(), best_bid)/best_bid,
                          np.subtract(books.iloc[:, 14:23].to_numpy(), best_ask)/best_ask], axis=1)
    S_V = books.iloc[:, 23:43].to_numpy()/mean_vol
    S_Lob = np.log(np.concatenate([S_P, S_V], axis=1))

    TC_IMBAL, TV_IMBAL, BEGIN_TC = tools.get_TCI_and_TVI(books, trades)
    mid_price_volatillity, BEGIN_MP = tools.get_midPrice_volatility(books)

    BEGIN = max(BEGIN_TC, BEGIN_MP, 101)
    if dy > begin_day:
        assert curr_idx > BEGIN
        BEGIN = curr_idx
    print('begin', BEGIN)

    FINAL_SIZE = len(books) - BEGIN
    END = -500
    FINAL_SIZE += END


    # OFI = ofi_features[BEGIN - 1:]
    # OFS = of_features[BEGIN - 1:]
    if END:
        timespan = books['time'].to_numpy()[BEGIN:END]
        mid_price = books['midPrice'].to_numpy()[BEGIN:END]
        bid1 = books['bid1'].to_numpy()[BEGIN:END]
        ask1 = books['ask1'].to_numpy()[BEGIN:END]
        OFI = ofi_features[BEGIN-1:END]
        OFS = of_features[BEGIN-1:END]
        LOB = books.iloc[:, 3:43][BEGIN:END].to_numpy()
        MPVO = np.array(mid_price_volatillity[BEGIN:END]).reshape(FINAL_SIZE, 1)
        BAI = bid_ask_imbs[BEGIN:END]
        PD = np.concatenate([np.subtract(best_bid, books.iloc[:, 4:13].to_numpy()),
                             np.subtract(books.iloc[:, 14:23].to_numpy(), best_ask)], axis=1)[BEGIN:END]
        TCI = np.array(TC_IMBAL[BEGIN:END]).reshape(FINAL_SIZE, 1)
        TVI = np.array(TV_IMBAL[BEGIN:END]).reshape(FINAL_SIZE, 1)
        WVPS = w_volume_price_spread[BEGIN:END].reshape(FINAL_SIZE, 1)
        HR = height_imb[BEGIN:END]
        # accumulated differences
        APD = np.sum(books.iloc[:, 13:23] - books.iloc[:, 3:13], axis=1)
        AVD = np.sum(books.iloc[:, 33:43] - books.iloc[:, 23:33], axis=1)
        AD = np.array([APD, AVD]).T[BEGIN:END]
        MPV = np.array([np.mean(books.iloc[:, 3:13], axis=1),
               np.mean(books.iloc[:, 13:23], axis=1),
               np.mean(books.iloc[:, 23:33], axis=1),
               np.mean(books.iloc[:, 33:43], axis=1),
               ]).T[BEGIN:END]

        PPRESS = price_press[BEGIN:END].reshape(FINAL_SIZE, 1)
        S_LOB = S_Lob[BEGIN:END]
        # print(len(S_LOB))
        label_100_3 = tools.get_class_label(books['midPrice'], 100, method=2)[BEGIN:END]
        label_50_3 = tools.get_class_label(books['midPrice'], 50, method=2)[BEGIN:END]
        label_20_3 = tools.get_class_label(books['midPrice'], 20, method=2)[BEGIN:END]
        label_70_3= tools.get_class_label(books['midPrice'], 70, method=2)[BEGIN:END]

        label_100_2 = tools.get_class_label(books['midPrice'], 100, method=1)[BEGIN:END]
        label_50_2 = tools.get_class_label(books['midPrice'], 50, method=1)[BEGIN:END]
        label_20_2 = tools.get_class_label(books['midPrice'], 20, method=1)[BEGIN:END]
        label_70_2 = tools.get_class_label(books['midPrice'], 70, method=1)[BEGIN:END]
    else:
        raise ValueError('reach the end')

    # all_features = [OFI, MPVO, BAI, PD, TCI, TVI, WVPS, HR, AD, LOB, MPV, OFS, PPRESS]
    # all_feature_names = ['OFI', 'MPVO', 'BAI', 'PD', 'TCI', 'TVI', 'WVPS', 'HR', 'AD', 'LOB',
    #                     'MPV', 'OFS', 'PPRESS']
    all_features = [S_LOB, LOB, OFI, OFS, BAI, HR, MPV, AD, TCI, TVI, MPVO, PPRESS, WVPS, PD]
    # all_features = [S_LOB, PD, BAI, HR, MPV, AD, TCI, TVI, MPVO, PPRESS, WVPS]
    # all_features = [BAI, HR, MPV, AD, TCI, TVI, MPVO, PPRESS, WVPS]
    print([len(i[0, :]) for i in all_features])
    print([len(i) for i in all_features])
    assert ( sum([len(i) - len(all_features[0]) for i in all_features]) == 0)
    all_feature_names = ['S_LOB', "LOB", 'OFI', "OFS", 'BAI', 'HR',  'MPV', 'AD', 'TCI', 'TVI', 'MPVO', 'PPRESS', 'WVPS', 'PD']
    # all_feature_names = ['S_LOB', 'PD', 'BAI', 'HR', 'MPV', 'AD', 'TCI', 'TVI', 'MPVO', 'PPRESS', 'WVPS']
    # all_feature_names = ['BAI', 'HR', 'MPV', 'AD', 'TCI', 'TVI', 'MPVO', 'PPRESS', 'WVPS']
    # "MPV", "AD", 'PPRESS', 'TCI', 'TVI', 'MPVO', 'WVPS']
    feature_list = defaultdict(list)
    feature_list['time'] = timespan
    feature_list['mid_price'] = mid_price
    feature_list['bid1'] = bid1
    feature_list['ask1'] = ask1
    count = 0
    for i in range(len(all_feature_names)):
        for j in range(np.shape(all_features[i])[1]):
            feature_list[all_feature_names[i] + '_' + str(j)] = all_features[i][:, j]
            count+=1
    print('num_fea', count)
    feature_list['label_20_3'] = label_20_3
    feature_list['label_50_3'] = label_50_3
    feature_list['label_70_3'] = label_70_3
    feature_list['label_100_3'] = label_100_3
    feature_list['label_20_2'] = label_20_2
    feature_list['label_50_2'] = label_50_2
    feature_list['label_70_2'] = label_70_2
    feature_list['label_100_2'] = label_100_2

    feature_df = pd.DataFrame(feature_list)
    _, month, day = book_files[dy].split('_')
    print(month, day)
    if END:
        prev_book = books[-(500+END):-END]
        prev_trade = trades[-(500+END):-END]
        print(len(prev_book))
    else:
        prev_book = books[-500:]
        prev_trade = trades[-500:]
    print(len(feature_df))
    feature_df.to_csv(target_dir + 'feature'+month+day, index=False)
