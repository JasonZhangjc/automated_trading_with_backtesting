"""
Channel Break-Out Detection
    - Pivot points detection
    - Channel dectection
    - Break-Out detection
Detect pivots, then fit two lines (upper = resistance, lower = support)
with those pivots to form a channel
Detect breakout points relative to the channel to indicate future trends
The intuition is based on Momentum

This module is for backtesting the breakout trading strategy

Extremely high return in backtesting
"""



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from backtesting import Strategy
from backtesting import Backtest

import breakout




def isPivot(candle, window):
    """
    function that detects if a candle is a pivot/fractal point
    args: candle index, window before and after candle to test if pivot
    returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
    """
    if candle-window < 0 or candle+window >= len(df):
        return 0

    pivotHigh = 1
    pivotLow = 2
    for i in range(candle-window, candle+window+1):
        if df.iloc[candle].Low > df.iloc[i].Low:
            pivotLow=0
        if df.iloc[candle].High < df.iloc[i].High:
            pivotHigh=0
    if (pivotHigh and pivotLow):
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0



def collect_channel(df, candle, back_candles, window):
    localdf = df[candle-back_candles-window:candle-window]
    #localdf['isPivot'] = localdf.apply(lambda x: isPivot(x.name,window), axis=1)
    highs = localdf[localdf['isPivot']==1].High.values
    idxhighs = localdf[localdf['isPivot']==1].High.index
    lows = localdf[localdf['isPivot']==2].Low.values
    idxlows = localdf[localdf['isPivot']==2].Low.index

    if len(lows)>=3 and len(highs)>=3:
        sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows,lows)
        sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs,highs)

        return(sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
    else:
        return(0,0,0,0,0,0)



def isBreakOut(df, candle, back_candles, window):
    if (candle-back_candles-window)<0:
        return 0

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = df.iloc[candle].Channel

    prev_idx = candle-1
    prev_high = df.iloc[candle-1].High
    prev_low = df.iloc[candle-1].Low
    prev_close = df.iloc[candle-1].Close

    curr_idx = candle
    curr_high = df.iloc[candle].High
    curr_low = df.iloc[candle].Low
    curr_close = df.iloc[candle].Close
    curr_open = df.iloc[candle].Open

    if ( prev_high > (sl_lows*prev_idx + interc_lows) and
        prev_close < (sl_lows*prev_idx + interc_lows) and
        curr_open < (sl_lows*curr_idx + interc_lows) and
        curr_close < (sl_lows*prev_idx + interc_lows)): #and r_sq_l > 0.9
        return 1

    elif ( prev_low < (sl_highs*prev_idx + interc_highs) and
        prev_close > (sl_highs*prev_idx + interc_highs) and
        curr_open > (sl_highs*curr_idx + interc_highs) and
        curr_close > (sl_highs*prev_idx + interc_highs)): #and r_sq_h > 0.9
        return 2

    else:
        return 0



def backtest(df, back_candles, window):
    df['isPivot'] = df.apply(lambda x: isPivot(x.name,window), axis=1)
    df['Channel'] = [collect_channel(df, candle, back_candles, window) for candle in df.index]
    df["isBreakOut"] = [isBreakOut(df, candle, back_candles, window) for candle in df.index]

    def SIGNAL():
        return df.isBreakOut

    class BreakOut(Strategy):
        initsize = 0.1
        mysize = initsize
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            TPSLRatio = 1.2

            if self.signal1==2 and len(self.trades)==0:
                sl1 = self.data.Low[-2]
                tp1 = self.data.Close[-1] + abs(self.data.Close[-1]-sl1)*TPSLRatio
                #tp2 = self.data.Close[-1] + abs(self.data.Close[-1]-sl1)*TPSLRatio/3
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
                #self.buy(sl=sl1, tp=tp2, size=self.mysize)

            elif self.signal1==1 and len(self.trades)==0:
                sl1 = self.data.High[-2]
                tp1 = self.data.Close[-1] - abs(sl1-self.data.Close[-1])*TPSLRatio
                #tp2 = self.data.Close[-1] - abs(sl1-self.data.Close[-1])*TPSLRatio/3
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
                #self.sell(sl=sl1, tp=tp2, size=self.mysize)

    bt = Backtest(df, BreakOut, cash=1000, margin=1/50, commission=.000)
    stat = bt.run()
    bt.plot()
    print(stat)
    return stat






if __name__ == "__main__":
    df = breakout.read_data_eurousd()

    back_candles = 45
    window = 3
    stat = backtest(df, back_candles, window)