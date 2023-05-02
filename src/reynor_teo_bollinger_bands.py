"""
Automated Rayner Teo Bollinger Bands Strategy Optimized For High Return
Extremely well
Simple strategy with Bollinger bands and fast & slow moving average
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
    dfSPY=yf.download("^RUI",start='2011-01-05', end='2021-01-05')
    dfSPY=dfSPY[dfSPY.High!=dfSPY.Low]
    dfSPY.reset_index(inplace=True)
    return dfSPY



def preprocess_data(dfSPY):
    # slow ema
    dfSPY['EMA']=ta.ema(dfSPY.Close, length=200)  #sma ema
    # faster ema
    dfSPY['EMA2']=ta.ema(dfSPY.Close, length=150) #sma ema
    dfSPY['RSI']=ta.rsi(dfSPY.Close, length=12)
    my_bbands = ta.bbands(dfSPY.Close, length=14, std=2.0)
    dfSPY=dfSPY.join(my_bbands)
    dfSPY.dropna(inplace=True)
    dfSPY.reset_index(inplace=True)
    return dfSPY



def add_ema_signal(df):
    emasignal = [0]*len(df)
    for i in range(0, len(df)):
        # if faster is above slow ema
        # it means a uptrend
        if df.EMA2[i] > df.EMA[i]:
            emasignal[i] = 2
        # if faster is below slow ema
        # it means a downtrend
        elif df.EMA2[i] < df.EMA[i]:
            emasignal[i] = 1
    df['EMASignal'] = emasignal
    return df



def add_orders_limit(df, percent):
    ordersignal=[0]*len(df)
    for i in range(1, len(df)): #EMASignal of previous candle!!! modified!!!
        # if close price is below the lower bbands and
        # we have a uptrend forecasting with ema
        # we will buy with a lower price than the close price
        if  df.Close[i]<=df['BBL_14_2.0'][i] and df.EMASignal[i]==2:
            ordersignal[i] = df.Close[i] - df.Close[i]*percent
        # if close price is above the upper bbands and
        # we have a downtrend forecasting with ema
        # we will sell with a higher price than the close price
        elif df.Close[i]>=df['BBU_14_2.0'][i] and df.EMASignal[i]==1:
            ordersignal[i] = df.Close[i] + df.Close[i]*percent
    df['ordersignal']=ordersignal
    return df



def add_point_pos_break(df):
    def get_point_pos_break(x):
        if x['ordersignal']!=0:
            return x['ordersignal']
        else:
            return np.nan
    df['pointposbreak'] = df.apply(lambda row: get_point_pos_break(row), axis=1)
    return df



def plot_signal(df, start, end):
    dfpl = df[start:end].copy()
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=2), name="EMA"),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA2, line=dict(color='yellow', width=2), name="EMA2"),        
                    go.Scatter(x=dfpl.index, y=dfpl['BBL_14_2.0'], line=dict(color='blue', width=1), name="BBL"),
                    go.Scatter(x=dfpl.index, y=dfpl['BBU_14_2.0'], line=dict(color='blue', width=1), name="BBU")])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                    marker=dict(size=6, color="MediumPurple"),
                    name="Signal")
    #fig.update(layout_yaxis_range = [300,420])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(autosize=False, width=600, height=600,margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="white")
    fig.show()



def backtest(df):
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
            TPSLRatio = 2
            perc = 0.02

            if len(self.trades)>0:
                if self.data.index[-1]-self.trades[-1].entry_time>=10:
                    self.trades[-1].close()
                if self.trades[-1].is_long and self.data.RSI[-1]>=75:
                    self.trades[-1].close()
                elif self.trades[-1].is_short and self.data.RSI[-1]<=25:
                    self.trades[-1].close()

            if self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==2:
                sl1 = min(self.data.Low[-1],self.data.Low[-2])*(1-perc)
                tp1 = self.data.Close[-1]+(self.data.Close[-1] - sl1)*TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)

            elif self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==1:
                sl1 = sl1 = max(self.data.High[-1],self.data.High[-2])*(1+perc)
                tp1 = self.data.Close[-1]-(sl1 - self.data.Close[-1])*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(dfpl, MyStrat, cash=1000, margin=1/5, commission=.000)
    stat = bt.run()
    bt.plot()
    print(stat)
    return stat





if __name__ == "__main__":
    df = read_data()
    df = preprocess_data(df)
    df = add_ema_signal(df)
    print(df.columns)

    percent = 0.000
    df = add_orders_limit(df, percent)
    df = add_point_pos_break(df)
    start, end = 1000, 3200
    plot_signal(df, 1000, 3200)
    stat = backtest(df)