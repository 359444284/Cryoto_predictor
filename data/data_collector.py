import os

import numpy as np
from cryptofeed import FeedHandler
# from cryptofeed.callback import
from cryptofeed.exchanges import Binance
from cryptofeed.defines import TRADES, L2_BOOK, BID, ASK
from decimal import Decimal

from qpython import qconnection
from qpython.qtype import QDOUBLE_LIST, QSTRING_LIST, QSYMBOL_LIST, QTIMESTAMP_LIST
import datetime
DEPTH = 10

q = qconnection.QConnection(host='localhost', port=5010, pandas=True)
q.open()
#
# q.sendSync("""trades:([]
#             date:`date$();
#             time:`time$();
#             symb:`symbol$();
#             side: `symbol$();
#             price: `float$();
#             volume: `float$()
#             )""")
#
# q.sendSync("""books:([]
#             date:`date$();
#             time:`time$();
#             symb:`symbol$();
#             bid1:`float$();
#             ask1:`float$();
#             bid2:`float$();
#             ask2:`float$();
#             bid3:`float$();
#             ask3:`float$();
#             bid4:`float$();
#             ask4:`float$();
#             bid5:`float$();
#             ask5:`float$();
#             bid6:`float$();
#             ask6:`float$();
#             bid7:`float$();
#             ask7:`float$();
#             bid8:`float$();
#             ask8:`float$();
#             bid9:`float$();
#             ask9:`float$();
#             bid10:`float$();
#             ask10:`float$();
#             bidVol1:`float$();
#             askVol1:`float$();
#             bidVol2:`float$();
#             askVol2:`float$();
#             bidVol3:`float$();
#             askVol3:`float$();
#             bidVol4:`float$();
#             askVol4:`float$();
#             bidVol5:`float$();
#             askVol5:`float$();
#             bidVol6:`float$();
#             askVol6:`float$();
#             bidVol7:`float$();
#             askVol7:`float$();
#             bidVol8:`float$();
#             askVol8:`float$();
#             bidVol9:`float$();
#             askVol9:`float$();
#             bidVol10:`float$();
#             askVol10:`float$()
#                         )""")


async def trade(t, receipt_timestamp):
    print(f"Trade received at {receipt_timestamp}", str(t.symbol),
        str(t.side),
        float(t.price),
        float(t.amount))

    date, time = str(datetime.datetime.fromtimestamp(t.timestamp).isoformat()).replace("-", ".").split('T')
    q.sendSync('`trades insert(`date${};`$\"{}\";`time${}; `{};`float${};`float${})'.format(
        date,
        str(t.symbol),
        time,
        str(t.side),
        float(t.price),
        float(t.amount)
    ))

async def books(book, receipt_timestamp):
    print('Feed: {} Pair: {} System Timestamp: {}'.format(
        book.exchange, book.symbol, receipt_timestamp
    ))
    ob = book.book

    date, time = str(datetime.datetime.fromtimestamp(receipt_timestamp).isoformat()).replace("-", ".").split('T')
    qstr = f"`books insert (`date${date}; `$\"{book.symbol}\"; `time${time}"
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
        f.add_feed(Binance(symbols=['BTC-USDT'], channels=[TRADES, L2_BOOK], max_depth=DEPTH ,callbacks={TRADES: trade, L2_BOOK: books}))
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
