import numpy as np
from qpython import qconnection
import datetime
from threading import Thread
import websocket
import json

q = qconnection.QConnection(host='localhost', port=5011, pandas=True)
q.open()

DEPTH = 10



def get_books_data(socket, pair):
    def on_close(ws):
        print('connect close')

    def on_error(ws, error):
        print("Error",error)

    def BOOK_handle(ws, message):
        message = json.loads(message)

        print('Pair: {} System Timestamp: {}'.format(
            pair, str(datetime.datetime.today())
        ))
        bid = np.array(message['bids'])
        ask = np.array(message['asks'])

        tp = str(datetime.datetime.today().isoformat()).replace("-", ".").replace("T", "D")
        qstr = f"`books insert (`timestamp${tp}; `$\"{pair}\""

        if len(bid) >= 10:
            for i in range(DEPTH):
                qstr += f"; `real${float(bid[i, 0])}"
            for i in range(DEPTH):
                qstr += f"; `real${float(ask[i, 0])}"
            for i in range(DEPTH):
                qstr += f"; `real${float(bid[i, 1])}"
            for i in range(DEPTH):
                qstr += f"; `real${float(ask[i, 1])}"

            qstr += ")"
            q.sendSync(qstr, param=None)
        else:
            print('not enough depth', len(bid))

    ws = websocket.WebSocketApp(socket, on_message=BOOK_handle,
                                on_close=on_close,
                                on_error=on_error)
    thread = Thread(target=ws.run_forever, args=(None, None, 60, 30))
    thread.start()


def get_trade_data(socket):
    def on_close(ws):
        print('connect close')

    def on_error(ws, error):
        print("Error", error)

    def trade_handler(ws, message):
        message = json.loads(message)
        print(f"Trade received at {message['T']}",
              str(message['s']),
              str(not message['m']),
              float(message['p']),
              float(message['q']))

        tp = str(datetime.datetime.fromtimestamp(message['T']/1000).isoformat()).replace("-", ".").replace("T", "D")
        q.sendSync('`trades insert(`timestamp${};`$\"{}\"; `boolean${};`real${};`real${})'.format(
            tp,
            str(message['s']),
            int(not message['m']),
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

    get_books_data(order_socket1, 'BTCUSDT')
    get_books_data(order_socket2, 'ETHUSDT')
    get_books_data(order_socket3, 'SOLUSDT')
    get_trade_data(trades_socket1)
    get_trade_data(trades_socket2)
    get_trade_data(trades_socket3)
