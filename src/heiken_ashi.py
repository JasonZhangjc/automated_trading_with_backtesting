"""
Heiken Ashi Candles
Simple rules to generate these candles
signal opportunities with fast and slow moving average
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest




def read_data():
    df = yf.download(tickers = '^RUI', start = '2012-03-11',end = '2022-07-10')
    df.drop(['Volume'], axis=1, inplace=True)
    df = df[df.Open != df.Close]
    df = df.reset_index()
    return df


def add_heiken_ashi(df):
    # use the mean value of today([open,close,high,low]) as heiken close price
    df['Heiken_Close'] = (df.Open+df.Close+df.High+df.Low)/4
    # use the average of yesterday([open,close]) as heiken open price
    df['Heiken_Open'] = df['Open']
    for i in range(1, len(df)):
        df['Heiken_Open'][i] = (df.Heiken_Open[i-1]+df.Heiken_Close[i-1])/2
    # use the max of [today([high]), heiken open, heiken close] as heiken high
    df['Heiken_High'] = df[['High', 'Heiken_Open', 'Heiken_Close']].max(axis=1)
    # use the min of [today([low]), heiken open, heiken close] as heiken high
    df['Heiken_Low'] = df[['Low', 'Heiken_Open', 'Heiken_Close']].min(axis=1)
    df.dropna(inplace=True)
    return df


def add_ref(df):
    # EMA20 is the fast moving average
    df["EMA20"] = ta.ema(df.Close, length=20)
    # EMA50 is the slow moving average
    df["EMA50"] = ta.ema(df.Close, length=50)
    df['RSI'] = ta.rsi(df.Close, length=12)
    return df


def plot_heiken_ashi(df, start=0, end=5000):
    dfpl = df[start:end]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Heiken_Open'],
                    high=dfpl['Heiken_High'],
                    low=dfpl['Heiken_Low'],
                    close=dfpl['Heiken_Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA20, line=dict(color='red', width=1), name="EMA20"),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA50, line=dict(color='blue', width=1), name="EMA50")])
    fig.show()


def total_signal(df):
    """
    Generate signal for trading opportunities with heiken ashi candles
    and fast and slow moving average lines
    2 for bullish
    1 for bearish
    """
    ordersignal=[0]*len(df)
    for i in range(0, len(df)):
        # if fast ma > slow ma means we are in a uptrend and
        # heiken open price < fast ma and
        # heiken close price > fast ma
        # bullish opportunity - buy in
        if (df.EMA20[i]>df.EMA50[i] and 
            df.Heiken_Open[i]<df.EMA20[i] and 
            df.Heiken_Close[i]>df.EMA20[i]):
            ordersignal[i]=2
        # if fast ma < slow ma means we are in a downtrend and
        # heiken open price > fast ma and
        # heiken close price < fast ma
        # bearish opportunity - sell out
        if (df.EMA20[i]<df.EMA50[i] and 
            df.Heiken_Open[i]>df.EMA20[i] and 
            df.Heiken_Close[i]<df.EMA20[i]):
            ordersignal[i]=1
    df['ordersignal']=ordersignal
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df


def add_point_pos(df):
    def get_point_pos(x):
        # bearish
        if x['ordersignal']==1:
            return x['High'] + 5
        # bullish
        elif x['ordersignal']==2:
            return x['Low'] - 5
        else:
            return np.nan
    df['pointpos'] = df.apply(lambda row: get_point_pos(row), axis=1)
    return df


def plot_signal_with_heiken_ashi(df, start=0, end=5000):
    dfpl = df[start:end]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Heiken_Open'],
                    high=dfpl['Heiken_High'],
                    low=dfpl['Heiken_Low'],
                    close=dfpl['Heiken_Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA20, line=dict(color='red', width=1), name="EMA20"),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA50, line=dict(color='blue', width=1), name="EMA50")])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=10, color="MediumPurple"),
                    name="Signal")
    fig.show()


def plot_signal_with_original_candle(df, start=0, end=5000):
    dfpl = df[start:end]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                        go.Scatter(x=dfpl.index, y=dfpl.EMA20, line=dict(color='red', width=1), name="EMA20"),
                        go.Scatter(x=dfpl.index, y=dfpl.EMA50, line=dict(color='blue', width=1), name="EMA50")])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=10, color="MediumPurple"),
                    name="Signal")
    fig.show()


def add_stoploss(df):
    # add StopLoss from signal
    SLSignal = [0] * len(df)
    SLbackcandles = 1
    for row in range(SLbackcandles, len(df)):
        mi =  1e10
        ma = -1e10
        # if bearish
        if df.ordersignal[row]==1:
            # max of the past high, can use other way
            for i in range(row-SLbackcandles, row+1):
                ma = max(ma,df.High[i])
            SLSignal[row]=ma
        # if bullish
        if df.ordersignal[row]==2:
            # min of the past low, can use other way
            for i in range(row-SLbackcandles, row+1):
                mi = min(mi,df.Low[i])
            SLSignal[row]=mi
    df['SLSignal']=SLSignal
    return df


def backtest(df):
    dfpl = df[:]

    def SIGNAL():
        return dfpl.ordersignal

    class MyStrat(Strategy):
        initsize = 0.02
        mysize = initsize
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            TPSLRatio = 1.5

            #if len(self.trades)>0:
            #    if self.trades[-1].is_long and self.data.RSI[-1]>=70:
            #        self.trades[-1].close()
            #    elif self.trades[-1].is_short and self.data.RSI[-1]<=30:
            #        self.trades[-1].close()

            if len(self.trades)>0:
                if (self.trades[-1].is_long
                    and self.data.Heiken_Open[-1]>=self.data.Heiken_Open[-1]):
                    self.trades[-1].close()
                elif (self.trades[-1].is_short
                    and self.data.Heiken_Open[-1]<=self.data.Heiken_Open[-1]):
                    self.trades[-1].close()

            if self.signal1==2 and len(self.trades)==0:
                sl1 = self.data.SLSignal[-1]
                tp1 = self.data.Close[-1]+(self.data.Close[-1] - sl1)*TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)

            elif self.signal1==1 and len(self.trades)==0:
                sl1 = self.data.SLSignal[-1]
                tp1 = self.data.Close[-1]-(sl1 - self.data.Close[-1])*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
    
    bt = Backtest(dfpl, MyStrat, cash=10000, margin=1/50, commission=.00)
    stat = bt.run()
    bt.plot(show_legend=False)
    print(stat)

    return stat






if __name__ == "__main__":
    df = read_data()
    df = add_heiken_ashi(df)
    df = add_ref(df)
    start, end = 2000, 2100
    plot_heiken_ashi(df, start, end)

    df = total_signal(df)
    df = add_point_pos(df)
    plot_signal_with_heiken_ashi(df, start, end)
    plot_signal_with_original_candle(df, start, end)

    df = add_stoploss(df)
    stat = backtest(df)