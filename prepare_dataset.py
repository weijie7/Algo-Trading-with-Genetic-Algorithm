#~* utf-8
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ta import add_all_ta_features
# from ta.utils import dropna
import pandas_ta as ta


#---- Default is fixed using ETH-USD, 15min, fixed date-range

def date2unix(date_time):
    return int(time.mktime(datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S').timetuple()))

def unix2date(unix):
    return datetime.utcfromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S')

def get_kucoin_hist(symbol, kline_type, start, end):
    total_hist = []

    hist = client.get_kline_data(symbol, kline_type, start, end) #end is not inclusive // order in descending
    total_hist.extend(hist)

    while int(hist[-1][0]) > start:
        end = int(hist[-1][0])
        hist = client.get_kline_data(symbol, kline_type, start, end) 
        total_hist.extend(hist)
        time.sleep(2)
    
    return total_hist

def HA(df):
    df['HA_Close_t']=(df['open']+ df['high']+ df['low']+df['close'])/4

    from collections import namedtuple
    nt = namedtuple('nt', ['open','close'])
    previous_row = nt(df.iloc[0]['open'],df.iloc[0]['close'])

    HA_open_t = []
    for row in df.itertuples():
        ha_open = (previous_row.open + previous_row.close) / 2
        HA_open_t.append(ha_open)
        previous_row = nt(ha_open, row.close)

    df['HA_Open_t'] = pd.Series(HA_open_t)
    df['HA_High_t']=df[['HA_Open_t','HA_Close_t','high']].max(axis=1)
    df['HA_Low_t']=df[['HA_Open_t','HA_Close_t','low']].min(axis=1)
    return df


def retrieve_data(file_dir= 'data/ETH_USDT-5m.json'):
    df = pd.read_json(file_dir)
    df.columns =  ['date','open','high','low','close','volume']
    return df

def generate_features(df):
    import pandas_ta as ta
    #rsi
    df['momentum_rsi'] = df.ta.rsi()
    
    #stochastic
    stoch_fast = ta.stochf(df['high'], df['low'], df['close'])
    df['fastd'] = stoch_fast['STOCHFd_14_3']
    df['fastk'] = stoch_fast['STOCHFk_14_3']

    #heikin ashi
    df = HA(df)
    df['heiken_trend'] = df.apply(lambda x: 0 if x['HA_Open_t'] == x['HA_High_t'] else (1 if x['HA_Open_t'] == x['HA_Low_t'] else 0.5 - (( (min(x['HA_Open_t'], x['HA_Close_t']) - x['HA_Low_t']) - (x['HA_High_t'] - max(x['HA_Open_t'], x['HA_Close_t'])) )/ ((min(x['HA_Open_t'], x['HA_Close_t']) - x['HA_Low_t']) + (x['HA_High_t']) - max(x['HA_Open_t'], x['HA_Close_t'])) )*0.5 ), axis=1 )
    df['heiken_trend_sm3d'] = df['heiken_trend'].rolling(3).mean()
    df['heiken_trend_sm4d'] = df['heiken_trend'].rolling(4).mean()
    df['heiken_trend_sm2d'] = df['heiken_trend'].rolling(2).mean()
    df.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'}, inplace = True)
    return df

def save_file(df):
    df.to_csv('data/all_feat_working.csv', index=None)


if __name__ == '__main__':
    df = retrieve_data()
    df = generate_features(df)
    save_file(df)
    print('Job done')



