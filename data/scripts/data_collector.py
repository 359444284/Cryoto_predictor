import argparse
import os
import logging
import numpy as np

from qpython import qconnection
import datetime
DEPTH = 10

q = qconnection.QConnection(host='localhost', port=5011, pandas=True)
q.open()
#


async def trade(t, receipt_timestamp):
    print(f"Trade received at {receipt_timestamp}", str(t.symbol),
        str(t.side),
        float(t.price),
        float(t.amount))

    tp = str(datetime.datetime.fromtimestamp(t.timestamp).isoformat()).replace("-", ".")
    q.sendSync('`trades insert(`timestamp${};`$\"{}\"; `{};`float${};`float${})'.format(
        tp,
        str(t.symbol),
        str(t.side),
        float(t.price),
        float(t.amount)
    ))

async def books(book, receipt_timestamp):
    print('Feed: {} Pair: {} System Timestamp: {}'.format(
        book.exchange, book.symbol, receipt_timestamp
    ))
    ob = book.book

    tp = str(datetime.datetime.fromtimestamp(receipt_timestamp).isoformat()).replace("-", ".")
    qstr = f"`books insert (`timestamp${tp}; `$\"{book.symbol}\""
    volumes = [] #[bid1, ask1, bid2, ask2, .....]
    if len(ob.bid) >= 10:
        for i in range(DEPTH):
            bid_p, bid_v = ob.bid.index(i)
            ask_p, ask_v = ob.ask.index(i)
            volumes.extend([float(bid_v), float(ask_v)])
            qstr += f";`float${float(bid_p)}; `float${float(ask_p)}"
        for i in range(0, DEPTH*2, 2):
            qstr += f";`float${volumes[i]}; `float${volumes[i+1]}"
        qstr += ")"
        q.sendSync(qstr, param=None)
    else:
        print('not enough depth', len(ob.bid))


def main():
    try:
        f = FeedHandler()
        f.add_feed(Binance(symbols=['BTC-USDT'], channels=[TRADES, L2_BOOK], max_depth=DEPTH
                           ,callbacks={TRADES: trade, L2_BOOK: books}))
        f.add_feed(Binance(symbols=['ETH-USDT'], channels=[TRADES, L2_BOOK], max_depth=DEPTH,
                           callbacks={TRADES: trade, L2_BOOK: books}))
        f.add_feed(Binance(symbols=['DOGE-BUSD'], channels=[TRADES, L2_BOOK], max_depth=DEPTH,
                           callbacks={TRADES: trade, L2_BOOK: books}))
        f.run()
    except KeyboardInterrupt:
        pass
    finally:
        # save trades and quotes tables to disk
        # data_path = os.getcwd()
        # data_path = data_path.replace("\\", "/")
        # trades_path = f"`:{data_path}/trades set trades"
        # quotes_path = f"`:{data_path}/quotes set quotes"
        # print(f"saving to disk quotes -> {quotes_path} trades -> {trades_path}")
        q.close()

if __name__=='__main__':
    main()
