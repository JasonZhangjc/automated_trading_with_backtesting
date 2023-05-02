"""
Rob Hoffman's Inventory Retracement Bar
Award winning strategy
Two main components:
    - Inventory Retracement Bar: IRB
    - Trend identification
"""



import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest





def read_data():
    # df = yf.download(tickers='EURUSD=X', period='60d', interval='15m')
    df = pd.read_csv("EURUSD_Candlestick_1_Hour_BID_05.05.2003-08.02.2022.csv")
    return df


def preprocessing(df, backrollingN = 20):
    df.reset_index(drop=True, inplace=True)
    df["EMA"] = ta.ema(df.Close, length=20)
    df["ATR"] = ta.atr(df.High, df.Low, df.Close, length=14)
    df['slopeEMA'] = df['EMA'].diff(periods=1)
    df['slopeEMA'] = df['slopeEMA'].rolling(window=backrollingN).mean()
    return df


def get_signal(df, slope_limit, percent_limit):
    TotSignal = [0] * len(df)
    for row in range(0, len(df)):
        # if the slope is negative enough and
        # (min(open, close)-low) / (high-low) > percent_limit
        # we find a IRB in the downtrend
        # we forecast a future downtrend
        if (df.slopeEMA[row] < -slope_limit and
            (min(df.Open[row], df.Close[row]) - df.Low[row]) / 
            (df.High[row]-df.Low[row]) 
             > percent_limit):
            TotSignal[row]=1
        # if the slope is positive enough and
        # (high-max(open, close)) / (high-low) > percent_limit
        # we find a IRB in the uptrend
        # we forecast a future uptrend
        if (df.slopeEMA[row] > slope_limit and
            (df.High[row] - max(df.Open[row], df.Close[row])) / 
            (df.High[row]-df.Low[row]) 
             > percent_limit):
            TotSignal[row]=2
    df['TotSignal']=TotSignal
    return df


def generate_signal(df, slopelimit, percentlimit,
                    left, right):
    """
    left: candles before the Hoffman candle
    right: candles after the Hoffman candle
    """
    TotSignal = [0] * len(df)
    # row-right is the candle where we determine whether it is IRB or not
    # row-left is the leftmost candle we look at
    # row is the rightmost candle we look at
    for row in range(left, len(df)):
        # if rwo-right is an IRB candle and
        # the IRB low is the valley
        if (df.slopeEMA[row-right] < -slopelimit and 
            (min(df.Open[row-right], df.Close[row-right])-df.Low[row-right]) /
            (df.High[row-right]-df.Low[row-right]) 
            > percentlimit and
            df.Low[row-right] <= df.Low[row-left : row].min()):
            # bearish
            TotSignal[row-right]=1
        # if rwo-right is an IRB candle and
        # the IRB high is the peak
        if (df.slopeEMA[row-right] > slopelimit and
            (df.High[row-right] - max(df.Open[row-right], df.Close[row-right])) / 
            (df.High[row-right]-df.Low[row-right]) 
            > percentlimit and
            df.High[row-right] >= df.High[row-left:row].max()):
            # bullish
            TotSignal[row-right]=2
    df['TotSignal']=TotSignal
    return df


def add_point_pos(df):
    def get_point_pos(x):
        if x['TotSignal']==1:
            return x['High']+1e-3
        elif x['TotSignal']==2:
            return x['Low']-1e-3
        else:
            return np.nan
    df['pointpos'] = df.apply(lambda row: get_point_pos(row), axis=1)
    return df


def plot_signal(df, start, end):
    dfpl = df[start:end]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=1), name="EMA")])
    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="Signal")
    fig.show()


def add_ema_signal(df, back_candles = 6):
    emasignal = [0]*len(df)
    for row in range(back_candles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-back_candles, row+1):
            if df.High[i] >= df.EMA[i]:
                dnt=0
            if df.Low[i] <= df.EMA[i]:
                upt=0
        if upt==1 and dnt==1:
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2    # uptrend
        elif dnt==1:
            emasignal[row]=1    # downtrend
    df['EMASignal'] = emasignal
    return df


def add_hoffman_break_signal(df, back_candles):
    def get_hoffman_break_signal(curr, back_candles):
        for r in range(curr-back_candles-1, curr):
            # if uptrend according to ema signal and
            # if uptrend according to IRB signal and
            # the close price of curr is greater than the high of past
            # return bullish signal
            # this is a buyin position
            if (df.EMASignal[curr] == 2 and 
                df.TotSignal[r] == 2 and 
                df.Close[curr] >= df.High[r]):
                return 2
            # if downtrend according to ema signal and
            # if downtrend according to IRB signal and
            # the close price of curr is smaller than the low of past
            # return bearish signal
            # this is a sellout signal
            elif (df.EMASignal[curr] == 1 and 
                  df.TotSignal[r] == 1 and 
                  df.Close[curr] <= df.Low[r]):
                return 1
    HoffmanBreak = [0]*len(df)
    #careful back_candles used previous cell
    for row in range(back_candles, len(df)):
        HoffmanBreak[row] = get_hoffman_break_signal(row, back_candles)
    df['HoffmanBreakSignal'] = HoffmanBreak
    return df


def add_point_pos_break(df):
    def get_point_pos_break(x):
        if x['HoffmanBreakSignal']==1:
            return x['High']+1e-3
        elif x['HoffmanBreakSignal']==2:
            return x['Low']-1e-3
        else:
            return np.nan
    df['pointposbreak'] = df.apply(lambda row: get_point_pos_break(row), axis=1)
    return df


def plot_break_signal(df, start=250, end=400):
    dfpl = df[start:end]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=1), name="EMA")])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="Signal")
    fig.show()


def add_stop_loss_signal(df, sl_back_candles):
    SLSignal = [0] * len(df)
    for row in range(sl_back_candles, len(df)):
        mi=1e10
        ma=-1e10
        if df.HoffmanBreakSignal[row]==1:
            for i in range(row-sl_back_candles, row+1):
                ma = max(ma,df.High[i])
            SLSignal[row]=ma
        if df.HoffmanBreakSignal[row]==2:
            for i in range(row-sl_back_candles, row+1):
                mi = min(mi,df.Low[i])
            SLSignal[row]=mi
    df['SLSignal']=SLSignal
    return df


def backtest1(df):
    dfpl = df[1800:2200]
    dfpl.dropna()

    def SIGNAL():
        return dfpl.HoffmanBreakSignal

    class MyStrat1(Strategy):
        initsize = 0.2
        mysize = initsize
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            slatr = 4*self.data.ATR[-1]
            TPSLRatio = 1.5

            if self.signal1==2 and len(self.trades)==0:
                sl1 = self.data.Close[-1] - slatr
                tp1 = self.data.Close[-1] + slatr*TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)

            elif self.signal1==1 and len(self.trades)==0:
                sl1 = self.data.Close[-1] + slatr
                tp1 = self.data.Close[-1] - slatr*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(dfpl, MyStrat1, cash=10000, margin=1/50, commission=.00)
    stat = bt.run()
    bt.plot()
    print(stat)
    return stat


# def backtest2(df):
#     dfpl = df[:]

#     def SIGNAL():
#         return dfpl.HoffmanBreakSignal

#     class MyStrat2(Strategy):
#         initsize = 0.2
#         mysize = initsize
#         def init(self):
#             super().init()
#             self.signal1 = self.I(SIGNAL)

#         def next(self):
#             super().next()
#             TPSLRatio = 1.5

#             if self.signal1==2 and len(self.trades)==0:
#                 sl1 = self.data.SLSignal[-1]
#                 tp1 = self.data.Close[-1]+(self.data.Close[-1] - sl1)*TPSLRatio
#                 self.buy(sl=sl1, tp=tp1, size=self.mysize)

#             elif self.signal1==1 and len(self.trades)==0:
#                 sl1 = self.data.SLSignal[-1]
#                 tp1 = self.data.Close[-1]-(sl1 - self.data.Close[-1])*TPSLRatio
#                 self.sell(sl=sl1, tp=tp1, size=self.mysize)

#     bt = Backtest(dfpl, MyStrat2, cash=10000, margin=1/50, commission=.00)
#     stat = bt.run()
#     bt.plot()
#     print(stat)
#     return stat




if __name__ == "__main__":
    df = read_data()
    print(df.columns)
    backrollingN = 20
    df = preprocessing(df, backrollingN)
    print(df.head())

    slope_limit = 5e-5
    percent_limit = 0.45
    df = get_signal(df, slope_limit, percent_limit)
    # slopelimit=5e-5
    # percentlimit = 0.45
    # left = 10
    # right = 5
    # df = generate_signal(df, slopelimit, percentlimit, left, right)
    df = add_point_pos(df)
    plot_signal(df, 1800, 2200)

    back_candles = 6
    df = add_ema_signal(df, back_candles)
    df = add_hoffman_break_signal(df, back_candles)
    df = add_point_pos_break(df)
    plot_break_signal(df, 1800, 2200)

    sl_back_candles = 4
    df = add_stop_loss_signal(df, sl_back_candles)

    stat1 = backtest1(df)
    # stat2 = backtest2(df)
