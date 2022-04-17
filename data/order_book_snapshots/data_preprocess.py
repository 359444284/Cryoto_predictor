import pandas as pd
import datetime


def stamp_to_date(stamp):
    dt = datetime.datetime.fromtimestamp(stamp / 1000)
    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]
    return formatted_time


book = pd.read_csv('BitMEX_XBTUSD_ob_10_2019_10_16.csv')
print(book)
book['data'] = book['date'].apply(stamp_to_date)
print(book['data'])