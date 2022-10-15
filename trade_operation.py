#~* utf-8
import requests
import pandas as pd
import json
import numpy as np
import time
from datetime import datetime
from ta import add_all_ta_features
import pandas_ta as ta
import os.path


from prepare_dataset import *

from backtesting.lib import SignalStrategy, TrailingStrategy, crossover, Strategy
from backtesting.test import SMA
from backtesting import Backtest, Strategy
from ta.utils import dropna
import pandas_ta as ta
from math import exp


# ---- Buy Sell Operation. Take in strategy and fitness_fn

def populate_entry_trend_old(df, metadata:list, metamask:list):
    ema_1, ema_2 = metadata[0], metadata[1]
    rsi_1, rsi_2 = metadata[2], metadata[3]
    rsi_val, fastk_value, fastd_value = metadata[4], metadata[5], metadata[6]

    # 7 buy rules
    buy_signal_array = list(zip(
        (df.ta.ema(ema_1) > df.ta.ema(ema_2)) & (df.ta.ema(ema_1).shift() <= df.ta.ema(ema_2).shift()), #rule1 ema range: ema1, ema2, enable_1
        (df.ta.rsi(rsi_1) > df.ta.rsi(rsi_2)) & (df.ta.rsi(rsi_1).shift() <= df.ta.rsi(rsi_2).shift()), #rule2 rsi crossover: rsi1, rsi2, enable_2
        df['momentum_rsi'] > rsi_val, #rule3 rsi >  value: rsi_val, enable_3
        (df['heiken_trend'] > df['heiken_trend_sm3d']) & (df['heiken_trend'].shift() <= df['heiken_trend_sm3d'].shift()), #rule4 heiken_trend > sm3d crossover
        df['fastk'] < fastk_value, #rule5 fastk < value
        df['fastd'] < fastd_value, #rule6 fastd < value
        (df['fastk'] > df['fastd']) & (df['fastk'].shift() <= df['fastd'].shift() ) #rule7, fastk crossover fastd
    ))

    result = np.sum(buy_signal_array*np.array(metamask), axis=1) == sum(metamask) 
    return result

#Function for buy and sell signal -- only long
def buy_sell_old(data, buy_signal, tp_target=0.10 , sl_target=-0.015, stake=500):
    trade_data = dict()
    trade_id = 0

    position = False 

    # init price
    SL_price = -1
    TP_price = 9999999

    for i in range(len(data)):
        if position:
            # close trade if meet SL / TP / sell_rules
            if (data['close'][i] >= TP_price) or (data['close'][i] <= SL_price): #or (data['close_rule'][i] == 1):
                #trade_data
                trade_data[trade_id]['close_price'] = data['close'][i]

                #init
                SL_price = -1
                TP_price = 9999999
                position = False 
            else:
                SL_price = max(SL_price, data['close'][i] + data['close'][i]*sl_target)


        elif buy_signal[i] == True: #no position, check if buy rule hit
            #trade_data
            trade_id +=1
            trade_data[trade_id] = dict()
            trade_data[trade_id]['action'] = 'long'
            trade_data[trade_id]['buy_price'] = data['close'][i]
            trade_data[trade_id]['close_price'] = np.nan
            trade_data[trade_id]['unit'] = stake/data['close'][i]

            
            SL_price = data['close'][i] + data['close'][i]*sl_target
            TP_price = data['close'][i] + data['close'][i]*tp_target

            position = True
        

    return trade_data


def populate_entry_trend(df, metadata:list, metamask:list):
    #long
    ema_1, ema_2 = metadata[0], metadata[1]
    rsi_1, rsi_2 = metadata[2], metadata[3]
    rsi_val, fastk_value, fastd_value = metadata[4], metadata[5], metadata[6]

    #short
    ema_1_s, ema_2_s = metadata[7], metadata[8]
    rsi_1_s, rsi_2_s = metadata[9], metadata[10]
    rsi_val_s, fastk_value_s, fastd_value_s = metadata[11], metadata[12], metadata[13]

    rules_cnt = 9

    # 9 buy rules
    buy_signal_array = list(zip(
        (df.ta.ema(ema_1) > df.ta.ema(ema_2)) & (df.ta.ema(ema_1).shift() <= df.ta.ema(ema_2).shift()), #rule1 ema range: ema1, ema2, enable_1
        (df.ta.rsi(rsi_1) > df.ta.rsi(rsi_2)) & (df.ta.rsi(rsi_1).shift() <= df.ta.rsi(rsi_2).shift()), #rule2 rsi crossover: rsi1, rsi2, enable_2
        df['momentum_rsi'] > rsi_val, #rule3 rsi >  value: rsi_val, enable_3
        (df['heiken_trend'] > df['heiken_trend_sm2d']) & (df['heiken_trend'].shift() <= df['heiken_trend_sm2d'].shift()), #rule4 heiken_trend > sm3d crossover
        (df['heiken_trend'] > df['heiken_trend_sm3d']) & (df['heiken_trend'].shift() <= df['heiken_trend_sm3d'].shift()), #rule4 heiken_trend > sm3d crossover
        (df['heiken_trend'] > df['heiken_trend_sm4d']) & (df['heiken_trend'].shift() <= df['heiken_trend_sm4d'].shift()), #rule4 heiken_trend > sm3d crossover
        df['fastk'] < fastk_value, #rule5 fastk < value
        df['fastd'] < fastd_value, #rule6 fastd < value
        (df['fastk'] > df['fastd']) & (df['fastk'].shift() <= df['fastd'].shift() ) #rule7, fastk crossover fastd
    ))

    # 9 short rules
    sell_signal_array = list(zip(
        (df.ta.ema(ema_1_s) < df.ta.ema(ema_2_s)) & (df.ta.ema(ema_1_s).shift() >= df.ta.ema(ema_2_s).shift()), #rule1 ema range: ema1, ema2, enable_1
        (df.ta.rsi(rsi_1_s) < df.ta.rsi(rsi_2_s)) & (df.ta.rsi(rsi_1_s).shift() >= df.ta.rsi(rsi_2_s).shift()), #rule2 rsi crossover: rsi1, rsi2, enable_2
        df['momentum_rsi'] < rsi_val_s, #rule3 rsi >  value: rsi_val, enable_3
        (df['heiken_trend'] < df['heiken_trend_sm2d']) & (df['heiken_trend'].shift() >= df['heiken_trend_sm2d'].shift()), #rule4 heiken_trend > sm3d crossover
        (df['heiken_trend'] < df['heiken_trend_sm3d']) & (df['heiken_trend'].shift() >= df['heiken_trend_sm3d'].shift()), #rule4 heiken_trend > sm3d crossover
        (df['heiken_trend'] < df['heiken_trend_sm4d']) & (df['heiken_trend'].shift() >= df['heiken_trend_sm4d'].shift()), #rule4 heiken_trend > sm3d crossover
        df['fastk'] > fastk_value_s, #rule5 fastk < value
        df['fastd'] > fastd_value_s, #rule6 fastd < value
        (df['fastk'] < df['fastd']) & (df['fastk'].shift() >= df['fastd'].shift() ) #rule7, fastk crossover fastd
    ))


    long_result = np.sum(buy_signal_array*np.array(metamask[:rules_cnt]), axis=1) == sum(metamask[:rules_cnt]) 
    short_result = np.sum(sell_signal_array*np.array(metamask[rules_cnt:]), axis=1) == sum(metamask[rules_cnt:]) 

    return long_result, short_result

def buy_sell_old2(data, long_signal, short_signal, tp_target=0.10 , sl_target=-0.015, stake=500):
    trade_data = dict()
    trade_id = 0
    long_position = False 
    short_position = False

    # init price
    SL_price = -1
    TP_price = 9999999
    short_SL_price = 9999999
    short_TP_price = -1

    for i in range(len(data)):
        '''
        if longposition:
            check if can close
            update trailing stop loss
            if long signal, close short and long
        elif shortposition:
            check if can close
            update trailing stop loss
            if short signal, close long and short
        #cant long and short at the same time
        elif data_long signal
            buy if hit
        elif data_short signal
            short if hit
        '''


        if long_position:
            # close trade if meet SL / TP / sell_rules
            if (data['close'][i] >= TP_price) or (data['close'][i] <= SL_price) or (short_signal[i] == 1):
                #trade_data
                trade_data[trade_id]['close_price'] = data['close'][i]
                trade_data[trade_id]['close_date'] = data['date'][i]

                #init
                SL_price = -1
                TP_price = 9999999
                long_position = False 
            else:
                SL_price = max(SL_price, data['close'][i] + data['close'][i]*sl_target)


        elif short_position:
            # close trade if meet SL / TP / sell_rules
            if (data['close'][i] <= short_TP_price) or (data['close'][i] >= short_SL_price) or (long_signal[i] == 1):
                #trade_data
                trade_data[trade_id]['close_price'] = data['close'][i]
                trade_data[trade_id]['close_date'] = data['date'][i]

                #init
                short_SL_price = 9999999
                short_TP_price = -1
                short_position = False 
            else:
                short_SL_price = min(short_SL_price, data['close'][i] - data['close'][i]*sl_target)



        if (long_signal[i] == 1) & (not long_position) & (not short_position) : #no position, check if buy rule hit
            #trade_data
            trade_id +=1
            trade_data[trade_id] = dict()
            trade_data[trade_id]['action'] = 'long'
            trade_data[trade_id]['buy_price'] = data['close'][i]
            trade_data[trade_id]['close_price'] = np.nan
            trade_data[trade_id]['unit'] = stake/data['close'][i]
            trade_data[trade_id]['open_date'] = data['date'][i]
            
            SL_price = data['close'][i] + data['close'][i]*sl_target
            TP_price = data['close'][i] + data['close'][i]*tp_target

            long_position = True
        
        elif (short_signal[i] == 1) & (not long_position) & (not short_position) : #no position, check if buy rule hit
            #trade_data
            trade_id +=1
            trade_data[trade_id] = dict()
            trade_data[trade_id]['action'] = 'short'
            trade_data[trade_id]['buy_price'] = data['close'][i]
            trade_data[trade_id]['close_price'] = np.nan
            trade_data[trade_id]['unit'] = stake/data['close'][i]
            trade_data[trade_id]['open_date'] = data['date'][i]
            
            SL_price = data['close'][i] - data['close'][i]*sl_target
            TP_price = data['close'][i] - data['close'][i]*tp_target

            short_position = True

    return trade_data


def profit_calc_report(trade_data, wallet_ori = 20000):
    long_short_map = dict(long=1, short=-1)
    tmp_df = pd.DataFrame.from_dict(trade_data, orient ='index').dropna()
    tmp_df['profit'] = tmp_df.apply(lambda x: (x['close_price'] - x['buy_price']) * x['unit'] * long_short_map[x['action']], axis=1)
    tmp_df['wallet'] = wallet_ori + tmp_df.profit.cumsum()
    tmp_df['Win'] = tmp_df.apply(lambda x: 1 if x['profit']*long_short_map[x['action']] > 0 else 0, axis=1)
    tmp_df['hodl_duration'] = tmp_df.apply(lambda x: (datetime.strptime(x['close_date'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(x['open_date'], '%Y-%m-%d %H:%M:%S')).total_seconds()/3600, axis=1)
    tmp_df['open_date_shift'] = tmp_df.open_date.shift()
    tmp_df['time_btwn_trade'] = tmp_df.apply(lambda x: '' if x['open_date_shift'] != x['open_date_shift'] else (datetime.strptime(x['open_date'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(x['open_date_shift'], '%Y-%m-%d %H:%M:%S')), axis=1)


    #print report
    print('=========')
    print('Number of trade:\t\t{}'.format(len(trade_data)))
    print('Average trade daily:\t\t{}'.format(round((df.unix.iloc[-1] - df.unix.iloc[0])/1000/(60*60*24) / len(trade_data))))
    print('\n')
    print('Net profit in USD:\t\t${:.2f}'.format(tmp_df.profit.cumsum().iloc[-1]))
    print('Net profit in %:\t\t{:.2f}%'.format(tmp_df.profit.cumsum().iloc[-1]/wallet_ori * 100))
    print('Win Rate %:\t\t\t{:.2f}%'.format(tmp_df['Win'].mean()*100))
    print('Compare to if HODL:\t\t{:.2f}%'.format((df['close'].iloc[-1]-df['close'].iloc[0])/df['close'].iloc[0]*100))
    print('\n')
    print('Max profit:\t\t\t${:.1f}'.format(max(tmp_df['profit'])))
    print('Max drawdown:\t\t\t${:.1f}'.format(min(tmp_df['profit'])))

    return tmp_df


def fitness_fn(trade_data, wallet_ori = 20000):
    if len(trade_data) == 0:
        return 0,0,0,0
    else:
        long_short_map = dict(long=1, short=-1)
        tmp_df = pd.DataFrame.from_dict(trade_data, orient ='index').dropna()
        tmp_df['profit'] = tmp_df.apply(lambda x: (x['close_price'] - x['buy_price']) * x['unit'] * long_short_map[x['action']], axis=1)
        tmp_df['Win'] = tmp_df.apply(lambda x: 1 if x['profit'] > 0 else 0, axis=1)

        #return trade_cnt, netprofit, netprofit%, winrate%
        return len(trade_data), tmp_df.profit.cumsum().iloc[-1], tmp_df.profit.cumsum().iloc[-1]/wallet_ori * 100, tmp_df['Win'].mean()*100


def SMA(array, n):
    """Simple moving average"""
    return pd.Series(array).rolling(n).mean()
   
def EMA(df, n):
    """Simple moving average"""
    return df.ta.ema(n)

def RSI(df, n):
    """Simple moving average"""
    return df.ta.rsi(n)


class WJStrategy(Strategy):
    # Ddeclare variable
    input_var = None
    sl_target = None

    def init(self):
        super().init()
        # process input
        self.metadata_buy = self.input_var[:8]
        self.metadata_sell = self.input_var[8:16]
        self.metamask_buy =  self.input_var[16:26]
        self.metamask_sell =  self.input_var[26:-1]
        self.tp_target = self.input_var[-1]/100

        # buy params
        self.buy_ema1, self.buy_ema2 = self.metadata_buy[0], self.metadata_buy[1]
        self.buy_rsi1, self.buy_rsi2 = self.metadata_buy[2], self.metadata_buy[3]
        self.buy_moment_rsi, self.buy_fastk, self.buy_fastd = self.metadata_buy[4], self.metadata_buy[5], self.metadata_buy[6]
        self.buy_ema_val = self.metadata_buy[7]

        self.buy_ema_co_rule = self.metamask_buy[0]
        self.buy_rsi_co_rule = self.metamask_buy[1]
        self.buy_moment_rsi_rule = self.metamask_buy[2]
        self.buy_ha_2d_rule = self.metamask_buy[3]
        self.buy_ha_3d_rule = self.metamask_buy[4]
        self.buy_ha_4d_rule = self.metamask_buy[5]
        self.buy_fastd_rule = self.metamask_buy[6]
        self.buy_fastk_rule = self.metamask_buy[7]
        self.buy_faskd_co_rule = self.metamask_buy[8]
        self.buy_ema_rule = self.metamask_buy[9]

        # sell params
        self.sell_ema1, self.sell_ema2 = self.metadata_sell[0], self.metadata_sell[1]
        self.sell_rsi1, self.sell_rsi2 = self.metadata_sell[2], self.metadata_sell[3]
        self.sell_moment_rsi, self.sell_fastk, self.sell_fastd = self.metadata_sell[4], self.metadata_sell[5], self.metadata_sell[6]
        self.sell_ema_val = self.metadata_sell[7]

        self.sell_ema_co_rule = self.metamask_sell[0]
        self.sell_rsi_co_rule = self.metamask_sell[1]
        self.sell_moment_rsi_rule = self.metamask_sell[2]
        self.sell_ha_2d_rule = self.metamask_sell[3]
        self.sell_ha_3d_rule = self.metamask_sell[4]
        self.sell_ha_4d_rule = self.metamask_sell[5]
        self.sell_fastd_rule = self.metamask_sell[6]
        self.sell_fastk_rule = self.metamask_sell[7]
        self.sell_faskd_co_rule = self.metamask_sell[8]
        self.sell_ema_rule = self.metamask_sell[9]
        
        
        # build DF
        self.ema1 = self.I(EMA, self.data.df, self.buy_ema1)
        self.ema2 = self.I(EMA, self.data.df, self.buy_ema2)
        self.rsi1 = self.I(RSI, self.data.df, self.buy_rsi1)
        self.rsi2 = self.I(RSI, self.data.df, self.buy_rsi2)
        self.ema3 = self.I(EMA, self.data.df, self.buy_ema_val)

        self.ema1_s = self.I(EMA, self.data.df, self.sell_ema1)
        self.ema2_s = self.I(EMA, self.data.df, self.sell_ema2)
        self.rsi1_s = self.I(RSI, self.data.df, self.sell_rsi1)
        self.rsi2_s = self.I(RSI, self.data.df, self.sell_rsi2)
        self.ema3_s = self.I(EMA, self.data.df, self.sell_ema_val)

        
    def next(self):
        super().next()
        current_price = self.data.Close[-1]
        conditions = []
        if self.buy_ema_co_rule:
            conditions.append( (self.ema1[-1] > self.ema2[-1]) & (self.ema1[-2] <= self.ema2[-2]) )
        if self.buy_rsi_co_rule:
            conditions.append( (self.rsi1[-1] > self.rsi2[-1]) & (self.rsi1[-2] <= self.rsi2[-2]) )
        if self.buy_moment_rsi_rule:
            conditions.append(self.data.momentum_rsi[-1] > self.buy_moment_rsi)
        if self.buy_ha_2d_rule:
            conditions.append( (self.data.heiken_trend[-1] > self.data.heiken_trend_sm2d[-1]) & (self.data.heiken_trend[-2] <= self.data.heiken_trend_sm2d[-2]) )
        if self.buy_ha_3d_rule:
            conditions.append( (self.data.heiken_trend[-1] > self.data.heiken_trend_sm3d[-1]) & (self.data.heiken_trend[-2] <= self.data.heiken_trend_sm3d[-2]) )
        if self.buy_ha_4d_rule:
            conditions.append( (self.data.heiken_trend[-1] > self.data.heiken_trend_sm4d[-1]) & (self.data.heiken_trend[-2] <= self.data.heiken_trend_sm4d[-2]) )
        if self.buy_fastd_rule:
            conditions.append(self.data.fastk[-1]> self.buy_fastk)
        if self.buy_fastk_rule:
            conditions.append(self.data.fastd[-1] > self.buy_fastd)
        if self.buy_faskd_co_rule:
            conditions.append( (self.data.fastk[-1] > self.data.fastd[-1]) & (self.data.fastk[-2] <= self.data.fastd[-2]) )
        if self.buy_ema_rule:
            conditions.append( current_price > self.ema3[-1])
        if len(conditions) == 0:
            conditions = [False]

        sell_conditions = []
        if self.sell_ema_co_rule:
            sell_conditions.append( (self.ema1_s[-1] < self.ema2_s[-1]) & (self.ema1_s[-2] >= self.ema2_s[-2]) )
        if self.sell_rsi_co_rule:
            sell_conditions.append( (self.rsi1_s[-1] < self.rsi2_s[-1]) & (self.rsi1_s[-2] >= self.rsi2_s[-2]) )
        if self.sell_moment_rsi_rule:
            sell_conditions.append(self.data.momentum_rsi[-1] < self.sell_moment_rsi)
        if self.sell_ha_2d_rule:
            sell_conditions.append( (self.data.heiken_trend[-1] < self.data.heiken_trend_sm2d[-1]) & (self.data.heiken_trend[-2] >= self.data.heiken_trend_sm2d[-2]) )
        if self.sell_ha_3d_rule:
            sell_conditions.append( (self.data.heiken_trend[-1] < self.data.heiken_trend_sm3d[-1]) & (self.data.heiken_trend[-2] >= self.data.heiken_trend_sm3d[-2]) )
        if self.sell_ha_4d_rule:
            sell_conditions.append( (self.data.heiken_trend[-1] < self.data.heiken_trend_sm4d[-1]) & (self.data.heiken_trend[-2] >= self.data.heiken_trend_sm4d[-2]) )
        if self.sell_fastd_rule:
            sell_conditions.append(self.data.fastk[-1]< self.sell_fastk)
        if self.sell_fastk_rule:
            sell_conditions.append(self.data.fastd[-1] < self.sell_fastd)
        if self.sell_faskd_co_rule:
            sell_conditions.append( (self.data.fastk[-1] < self.data.fastd[-1]) & (self.data.fastk[-2] >= self.data.fastd[-2]) )
        if self.sell_ema_rule:
            sell_conditions.append( current_price < self.ema3_s[-1])

        if len(sell_conditions) == 0:
            sell_conditions = [False]

        if (not self.position) & (min(conditions)): 
            #buy signal + no position
            long_sl = current_price*(1-self.sl_target)
            long_tp = current_price*(1+self.tp_target)
            size = round(3000/current_price)
            self.buy(size = size, tp= long_tp, sl=long_sl)
            #print('buy', self.data.Close[-1], long_sl, long_tp)

        if (self.position.is_long) & (min(sell_conditions)):
            #sell signal + existing long
            self.position.close()
        
        if (not self.position) & (min(sell_conditions)): 
            #sell signal + no position
            short_sl = current_price*(1+self.sl_target)
            short_tp = current_price*(1-self.tp_target)
            size = round(3000/current_price)
            self.sell(size = size, tp= short_tp, sl=short_sl)
            #print('sell', self.data.Close[-1], short_sl, short_tp)

        if (self.position.is_short) & (min(conditions)):
            #buy signal + existing long
            self.position.close()



def backtesting(df, gene, sl_target, wallet=10000, commission=.006, trade_on_close=True):
    bt = Backtest(df, WJStrategy,
              cash=wallet, commission=commission,
              trade_on_close=trade_on_close)

    output = bt.run(input_var = gene, sl_target=sl_target)

    #fitness fn
    '''
    trade loss - number of trades. optimumly, we want each month period to be 20-30 trades
    profit loss - at least 100% return for 9 months
    duration loss - max trade duration, probably 60 min on average
    '''

    #full data
    if len(output['_trades']) == 0 :
        full_data_loss = 0
        recent_data_loss = -10
    else:
        trade_duration = output['_trades'].Duration.mean().seconds/60
        
        trade_loss = 0.2 * exp(-(output[17] - 500) ** 2 / 10 ** 5.8)
        profit_loss =  2*output[6] / 100
        duration_loss = 1- (0.2 * (trade_duration / 60))
        full_data_loss = min(2*profit_loss, trade_loss + 2*profit_loss + duration_loss)
        #print(trade_loss, profit_loss, duration_loss)

        #recent
        recent_data = output['_trades'][ output['_trades'].EntryTime > '2022-09-01' ]
        if len(recent_data) > 3 :
            trade_loss = 0.2 * exp(-(len(recent_data) - 30) ** 2 / 10 ** 5.8)
            profit_loss = 2*recent_data.PnL.sum()/wallet / 10
            duration_loss = 1- (0.2 * ((recent_data.Duration.mean().seconds/60)/ 60))
            recent_data_loss = trade_loss + 2*profit_loss + duration_loss
            recent_data_loss = min(2*profit_loss, trade_loss + 2*profit_loss + duration_loss)
            #print(trade_loss, profit_loss, duration_loss)
        else:
            recent_data_loss = -10

    full_fitness_score = (full_data_loss + recent_data_loss)/2

    return full_fitness_score




