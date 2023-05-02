"""
Composite RSI Strategy for Long Only
    - RSI to indicate overbought or oversell
    - ADX to indicate the strength of the trend
    - moving average to indicate bullish or bearish

Works very well!
"""



import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest



def read_data():
    df=yf.download("^RUI",start='2011-01-05', end='2021-01-05')
    df=df[df.High!=df.Low]
    df.reset_index(inplace=True)
    return df


def preprocess_data(df):
    df['EMA']=ta.ema(df.Close, length=200)#sma ema
    df['RSI']=ta.rsi(df.Close, length=2)
    a=ta.adx(df.High, df.Low, df.Close, length=14)
    df=df.join(a.ADX_14)
    #df.ta.indicators()
    #help(ta.adx)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df


def add_ema_signal(df, back_candles):
    emasignal = [0]*len(df)
    for row in range(back_candles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-back_candles, row+1):
            if df.High[i]>=df.EMA[i]:
                dnt=0
            if df.Low[i]<=df.EMA[i]:
                upt=0
        if upt==1 and dnt==1:
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2
        elif dnt==1:
            emasignal[row]=1
    df['EMASignal'] = emasignal
    return df


def total_signal(df, ADX):
    ordersignal=[0]*len(df)
    for i in range(0, len(df)):
        # long if the following conditions are met
        # RSI value <= 25: oversold, can buy in
        # ADX value >= 20: trend is not weak
        # moving average signal tells us to buy in
        if (df.RSI[i]<=25 and
            ADX[i]>=20 and
            df.EMASignal[i]==2):
            ordersignal[i]=2
    df['ordersignal']=ordersignal
    return df


def add_point_pos(df):
    def get_point_pos(x):
        if x['ordersignal']==1:
            return x['High']+2e-3
        elif x['ordersignal']==2:
            return x['Low']-2e-3
        else:
            return np.nan
    df['pointpos'] = df.apply(lambda row: get_point_pos(row), axis=1)
    return df


def plot_signal(df, start, end):
    dfpl = df[start:end].copy()
    #dfpl=dfpl.drop(columns=['level_0'])#!!!!!!!!!!
    #dfpl.reset_index(inplace=True)
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=2), name="EMA")])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=6, color="MediumPurple"),
                    name="Signal")
    #fig.update(layout_yaxis_range = [300,420])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(autosize=False, width=600, height=600,margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="white")
    fig.show()


def backtest(df):
    """
    Performance is great!
    """
    dfpl = df[:].copy()

    def SIGNAL():
        return dfpl.ordersignal

    class MyStrat(Strategy):
        initsize = 0.99
        mysize = initsize

        def init(self):
            super().init()
            self.signal = self.I(SIGNAL)

        def next(self):
            super().next()
            # percentage for the stop loss
            # take the 0.98=1-perc of te lowest price
            perc=0.02
            if len(self.trades)>0:
                if self.data.index[-1]-self.trades[-1].entry_time>=1000:
                    self.trades[-1].close()
                if self.trades[-1].is_long and self.data.RSI[-1]>=75:
                    self.trades[-1].close()
                elif self.trades[-1].is_short and self.data.RSI[-1]<=25:
                    self.trades[-1].close()
            if self.signal==2 and len(self.trades)==0: 
                sl1 = min(self.data.Low[-1],self.data.Low[-2])*(1-perc)
                #tp1 = self.data.Close[-1]+(self.data.Close[-1] - sl1)*TPSLRatio
                #self.buy(sl=sl1, tp=tp1, size=self.mysize)
                self.buy(sl=sl1,size=self.mysize)

    bt = Backtest(dfpl, MyStrat, cash=1000, margin=1/5, commission=.000)
    stat = bt.run()
    bt.plot()
    print(stat)

    return stat




if __name__ == "__main__":
    df = read_data()
    df = preprocess_data(df)
    back_candles = 6
    df = add_ema_signal(df, back_candles)
    df = total_signal(df, df.ADX_14)
    df = add_point_pos(df)
    plot_signal(df, 2000, 3200)
    stat = backtest(df)