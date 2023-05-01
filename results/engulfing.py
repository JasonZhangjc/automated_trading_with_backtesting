"""
Use Engulfing candles as trade signal
"""



import yfinance as yf         # Yahoo! Finance API
import pandas as pd
import pandas_ta as ta
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest
import backtesting



def read_data_eurousd():
    """
    Obtain the EURO to USD exchange rate data for two years
    """
    df = yf.download("EURUSD=X", start="2021-05-11", end="2023-4-30", interval='1h')
    return df


def apply_bollinger_bands(df):
    """
    Apply bbands to the input df
    """
    # Bollinger Bands
    my_bbands = ta.bbands(df.Close, length=30, std=1.5)
    df=df.join(my_bbands)
    return df


def bollinger_engulfing_signal(df):
    """
    Use two day's candles: today and yesterday
    signaling bullish doji or berish doji or no
    with Bollinger Bands
    if a bullish Engulfing pattern is found below the Bollinger Bands, buy
    if a bearish Engulfing pattern is found above the Bollinger Bands, sell
    """
    #bullish signal
    # if today's close price is smaller than the lower bound
    # and today's close price is greater than today's open price
    # and yesterday's close price is smaller than yesterday's open price
    # and today's open price is smaller than yesterday's close price
    # and today's close price is greater than yesterday's open price
    if ( # df.Close.iloc[-1] < df['BBL_30_1.5'].iloc[-1] and
        # if the price is above the lower bound and below medium band
        # this is less strict than the first requirement
       df.Close.iloc[-1] > df['BBL_30_1.5'].iloc[-1] and
       df.Close.iloc[-1] < df['BBM_30_1.5'].iloc[-1] and
       df.Close.iloc[-1] > df.Open.iloc[-1] and
       df.Close.iloc[-2] < df.Open.iloc[-2] and
       df.Open.iloc[-1] < df.Close.iloc[-2] and
       df.Close.iloc[-1] > df.Open.iloc[-2] ):
        return 2

    #bearish signal
    # if today's close price is greater than the upper bound
    # and today's close price is smaller than today's open price
    # and yesterday's close price is greater than yesterday's open price
    # and today's open price is greater than yesterday's close price
    # and today's close price is smaller than yesterday's open price
    elif ( # df.Close.iloc[-1] > df['BBU_30_1.5'].iloc[-1] and
        # if the price is below the upper bound and above medium band
        # this is less strict than the first requirement
       df.Close.iloc[-1] < df['BBU_30_1.5'].iloc[-1] and
       df.Close.iloc[-1] > df['BBM_30_1.5'].iloc[-1] and
       df.Close.iloc[-1] < df.Open.iloc[-1] and
       df.Close.iloc[-2] > df.Open.iloc[-2] and
       df.Open.iloc[-1] > df.Close.iloc[-2] and
       df.Close.iloc[-1] < df.Open.iloc[-2] ):
         return 1

    #nosignal
    else:
        return 0


def add_bbands_engulfing(df):
    """
    add the bbands_engulfing signal to the df
    """
    signal = [0]*len(df)
    for i in tqdm(range(20,len(df))):
        # only consider candles of today, yesterday, the day before yesterday
        df_ = df[i-3:i+1]
        signal[i]= bollinger_engulfing_signal(df_)
    df["bollinger_engulfing_signal"] = signal
    return df


def get_point_pos_engulfing(df):
    """
    get point position on the graph drawn later
    """
    # if bearish engulfing
    if df['bollinger_engulfing_signal']==1:
        return df['High']+0.5e-3
    # if bullish engulfing
    elif df['bollinger_engulfing_signal']==2:
        return df['Low']-0.5e-3
    else:
        return np.nan


def apply_point_pos_engulfing(df):
    """
    apply point position on the graph to the df by calling get_point_pos(df)
    """
    df['pointpos'] = df.apply(lambda row: get_point_pos_engulfing(row), axis=1)
    df.reset_index(inplace=True)
    return df


def plot_bbands_engulfing(df):
    """
    plot the candles with Bollinger bands
    """
    st = 700
    dfpl = df[st:st+250].copy()
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl['BBL_30_1.5'], line=dict(color='blue', width=1), name="BBL"),
                    go.Scatter(x=dfpl.index, y=dfpl['BBU_30_1.5'], line=dict(color='blue', width=1), name="BBU")])

    # use a purple dot to indicate a signal! Can be bullish or berish 
    # depending on its position relative to the bbands
    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="Signal")
    fig.update_layout(autosize=False, width=1000, height=600)
    fig.show()


def break_out_strategy_engulfing(df):
    """
    apply break out strategy engulfing
    """
    def SIGNAL():
        return df.bollinger_engulfing_signal

    class BreakOut(Strategy):
        initsize = 0.5
        mysize = initsize
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            TPSLRatio = 1.5

            if self.signal1==2 and len(self.trades)==0:
                sl1 = min(self.data.Low[-2], self.data.Low[-1])
                tp1 = self.data.Close[-1] + abs(self.data.Close[-1]-sl1)*TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)

            elif self.signal1==1 and len(self.trades)==0:
                sl1 = max(self.data.High[-2], self.data.High[-1])
                tp1 = self.data.Close[-1] - abs(sl1-self.data.Close[-1])*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(df, BreakOut, cash=1000, margin=1/10)
    stat = bt.run()
    bt.plot()
    print(stat)
    return stat





if __name__ == "__main__":
    df = read_data_eurousd()
    df = apply_bollinger_bands(df)
    df = add_bbands_engulfing(df)
    df = apply_point_pos_engulfing(df)
    # print(df.columns)
    plot_bbands_engulfing(df)
    stat = break_out_strategy_engulfing(df)