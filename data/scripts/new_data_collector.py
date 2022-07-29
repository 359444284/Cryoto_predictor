import numpy as np
from qpython import qconnection
import datetime
from threading import Thread
import websocket

q = qconnection.QConnection(host='localhost', port=5011, pandas=True)
q.open()

DEPTH = 10

def on_close(ws):
    print('connect close')

def on_error(ws,error):
    print('connect close')

def get_books_data(socket, pair):

    def BOOK_handle(ws, message):
        print('Pair: {} System Timestamp: {}'.format(
            pair, datetime.datetime.timestamp()
        ))
        bid = np.array(message['bids'])
        print(bid)
        ask = np.array(message['asks'])
        print(ask)

        tp = str(datetime.datetime.today().isoformat()).replace("-", ".").replace("T", "D")
        qstr = f"`books insert (`timestamp${tp}; `$\"{pair}\""
        if len(bid) >= 10:
            for i in range(DEPTH):
                qstr += f";`real${float(bid[i, 0])};"
            for i in range(DEPTH):
                qstr += f";`real${float(ask[i, 0])};"
            for i in range(DEPTH):
                qstr += f";`real${float(bid[i, 1])};"
            for i in range(DEPTH):
                qstr += f";`real${float(ask[i, 1])};"

            qstr += ")"
            q.sendSync(qstr, param=None)
        else:
            print('not enough depth', len(bid))

    ws = websocket.WebSocketApp(socket, on_message=BOOK_handle,
                                on_close=on_close,
                                on_error=on_error)
    thread = Thread(target=ws.run_forever, args=(None, None, 60, 15))
    thread.start()

def get_trade_data(socket):

    def trade_handler(ws, message):
        print(f"Trade received at {message['t']}",
              str(message['s']),
              str(not message['m']),
              float(message['p']),
              float(message['q']))

        tp = str(datetime.datetime.fromtimestamp(message['t']).isoformat()).replace("-", ".").replace("T", "D")
        q.sendSync('`trades insert(`timestamp${};`$\"{}\"; `boolean${};`float${};`float${})'.format(
            tp,
            str(message['s']),
            bool(not message['m']),
            float(message['p']),
            float(message['q'])
        ))

    ws = websocket.WebSocketApp(socket, on_message=trade_handler,
                                on_close=on_close,
                                on_error=on_error)
    thread = Thread(target=ws.run_forever, args=(None, None, 60, 15))
    thread.start()




if __name__ == '__main__':
    order_socket1 = 'wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms'
    order_socket2 = 'wss://stream.binance.com:9443/ws/ethusdt@depth10@100ms'
    order_socket3 = 'wss://stream.binance.com:9443/ws/solusdt@depth10@100ms'
    trades_socket1 = 'wss://stream.binance.com:9443/ws/btcusdt@aggTrade'
    trades_socket2 = 'wss://stream.binance.com:9443/ws/ethusdt@aggTrade'
    trades_socket3 = 'wss://stream.binance.com:9443/ws/solusdt@aggTrade'

    try:
        get_books_data(order_socket1, 'BTCUSDT')
        get_books_data(order_socket2, 'ETHUSDT')
        get_books_data(order_socket3, 'SOLUSDT')
        get_trade_data(order_socket1)
        get_trade_data(order_socket2)
        get_trade_data(order_socket3)
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