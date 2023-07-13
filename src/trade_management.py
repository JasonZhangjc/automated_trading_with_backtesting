# Trade management
# Bollinger lines
# Mean reversion strategy
# Two trades strategy with discretely adaptive stoploss



import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest



def read_data():
    '''
    read data from .csv in data folder
    '''
    # S&P500, daily data
    dfSPY = pd.read_csv("..\\data\\SPY.USUSD_Candlestick_1_D_BID_16.02.2017-21.05.2022.csv")
    dfSPY = dfSPY[dfSPY.High!=dfSPY.Low]
    dfSPY = dfSPY.reset_index()
    # use pandas_ta to get sma and rsi
    dfSPY['EMA'] = ta.sma(dfSPY.Close, length=200) # sma ema
    dfSPY['RSI'] = ta.rsi(dfSPY.Close, length=2)
    # use pandas_ta to get Bollinger bands
    my_bbands = ta.bbands(dfSPY.Close, length=20, std=2.5)
    dfSPY = dfSPY.join(my_bbands)
    dfSPY.dropna(inplace=True)
    dfSPY.reset_index(inplace=True)
    return dfSPY



def add_ema_signal(df, backcandles):
    '''
    add <exponential moving average> signal 
    An exponential moving average (EMA) is a type of moving average (MA) that 
    places a greater weight and significance on the most recent data points.
    '''
    emasignal = [0]*len(df)
    for row in range(backcandles, len(df)):
        upt = 1     # up trend
        dnt = 1     # down trend
        for i in range(row-backcandles, row+1):
            # if high price >= ema, not a down trend
            if df.High[i]>=df.EMA[i]:
                dnt=0
            # if low price <= ema, not a up trend
            if df.Low[i]<=df.EMA[i]:
                upt=0
        # if both up and down trends, undetermined signal ...
        if upt==1 and dnt==1:
            #print("!!!!! check trend loop !!!!")
            emasignal[row]=3
        # 2 to indicate up trend
        elif upt==1:
            emasignal[row]=2
        # 1 to indicate down trend
        elif dnt==1:
            emasignal[row]=1
    df['EMASignal'] = emasignal
    return df



def add_orders_limit(df, percent):
    '''
    ordersignal means the timing or ordering/trading
    if buy, wait a low point to trade
    if sell, wait a high point to trade
    '''
    ordersignal=[0]*len(df)
    for i in range(1, len(df)): #EMASignal of previous candle!!! modified!!!
        # if up trend and 
        # if close price <= Bolinger band lower line
        # buy signal
        if df.EMASignal[i] == 2 and \
           df.Close[i] <= df['BBL_20_2.5'][i]:
           # and df.RSI[i]<=100: #Added RSI condition to avoid direct close condition
            # when the price is lower than (1-percent)*close, buy in
            ordersignal[i] = df.Close[i] - df.Close[i]*percent
        # if down trend and
        # if close price >= Bollinger band upper line
        # sell signal
        elif df.EMASignal[i] == 1 and \
             df.Close[i] >= df['BBU_20_2.5'][i]:
             # and df.RSI[i]>=0:
            # when the price is greater than (1+percent)*close, sell out
            ordersignal[i] = df.Close[i] + df.Close[i]*percent
    df['ordersignal'] = ordersignal
    return df



def add_pointpos(df):
    def pointpos_break(x):
        if x['ordersignal']!=0:
            return x['ordersignal']
        else:
            return np.nan
    df['pointposbreak'] = df.apply(lambda row: pointpos_break(row), axis=1)
    return df



def plot_signal(df):
    dfpl = df[1000:1250].copy()
    #dfpl=dfpl.drop(columns=['level_0'])#!!!!!!!!!!
    #dfpl.reset_index(inplace=True)
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA, 
                               line=dict(color='orange', width=2), name="EMA"),
                    go.Scatter(x=dfpl.index, y=dfpl['BBL_20_2.5'], 
                               line=dict(color='blue', width=1), name="BBL_20_2.5"),
                    go.Scatter(x=dfpl.index, y=dfpl['BBU_20_2.5'], 
                               line=dict(color='blue', width=1), name="BBU_20_2.5")])
    fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="Signal")
    fig.show()



def backtest_simple(df):
    '''
    A relatively simple trade management strategy
    '''
    dfpl = df[:].copy()

    def SIGNAL():
        return dfpl.ordersignal

    class MyStrat(Strategy):
        # initial position size is 5% of our equity
        initsize = 0.05
        ordertime=[]

        def init(self):
            super().init()
            self.signal = self.I(SIGNAL)

        def next(self):
            super().next()
            for j in range(0, len(self.orders)):
                # 10 days max to fulfill the order!!!
                if self.data.index[-1] - self.ordertime[0] > 10:
                    self.orders[0].cancel()
                    self.ordertime.pop(0)
            # determine when to close trades to TP or SL
            if len(self.trades) > 0:
                # if 10 days already
                if self.data.index[-1] - self.trades[-1].entry_time >= 10:
                    self.trades[-1].close()
                    #print(self.data.index[-1], self.trades[-1].entry_time)
                # if long position and RSI>=50, close open positions
                if self.trades[-1].is_long and self.data.RSI[-1]>=50:
                    self.trades[-1].close()
                # if short position and RSI<=50,close short positions
                elif self.trades[-1].is_short and self.data.RSI[-1]<=50:
                    self.trades[-1].close()
            # buyin
            if self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==2:
                # Cancel previous orders
                for j in range(0, len(self.orders)):
                    self.orders[0].cancel()
                    self.ordertime.pop(0)
                # Add new replacement order 
                # when buyin, stop loss is 1/2 of the signal price
                self.buy(sl=self.signal/2, limit=self.signal, size=self.initsize)
                self.ordertime.append(self.data.index[-1])
            # sellout
            elif self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==1:
                # Cancel previous orders
                for j in range(0, len(self.orders)):
                    self.orders[0].cancel()
                    self.ordertime.pop(0)
                # Add new replacement order
                # when sellout, stop loss is 2 of the signal price
                self.sell(sl=self.signal*2, limit=self.signal, size=self.initsize)
                self.ordertime.append(self.data.index[-1])
    
    # margin controls the leverage
    # 1/100 is a higher leverage than 1/10
    bt = Backtest(dfpl, MyStrat, cash=10000, margin=1/100, commission=.000)
    stat = bt.run()
    bt.plot(show_legend=False)
    print(stat._trades)
    print(stat)
    return stat



def backtest_worse(df):
    '''
    A relatively worse trade management strategy
    '''
    dfpl = df[:].copy()

    def SIGNAL():
        return dfpl.ordersignal

    class MyStrat(Strategy):
        mysize = 0.05 #1000
        def init(self):
            super().init()
            self.signal = self.I(SIGNAL)

        def next(self):
            super().next()
            TPSLRatio = 2.
            perc = 0.02
            # when buyin, 
            if self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==2:
                # if price is as low as 0.98*close_price, we stop loss
                sl1 = self.data.Close[-1] - self.data.Close[-1]*perc
                sldiff = abs(sl1-self.data.Close[-1])
                # two TP, two buy
                # separate long positions into two slices
                # tp2 is triggered earlier than tp1
                # tp2 earns potentially less than tp1
                tp1 = self.data.Close[-1]+sldiff*TPSLRatio
                tp2 = self.data.Close[-1]+sldiff
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
                self.buy(sl=sl1, tp=tp2, size=self.mysize)
            # when sellout,
            elif self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==1:
                # if price is as high as 1.02*close_price, we stop loss
                sl1 = self.data.Close[-1] + self.data.Close[-1]*perc
                sldiff = abs(sl1-self.data.Close[-1])
                # two TP, two sell
                # separate short positions into two slices
                # tp2 is triggered earlier than tp1
                # tp2 earns potentially less than tp1
                tp1 = self.data.Close[-1]-sldiff*TPSLRatio
                tp2 = self.data.Close[-1]-sldiff
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
    
    # margin controls the leverage
    # 1/100 is a higher leverage than 1/10
    bt = Backtest(dfpl, MyStrat, cash=10000, margin=1/100, commission=.000)
    stat = bt.run()
    bt.plot(show_legend=False)
    print(stat._trades)
    print(stat)
    return stat



def backtest_better(df):
    '''
    A relatively better trade management strategy
    Two trades with discretely adaptive stoploss
    However, the results are not very good
    But, it can change with different data and parameters :)
    '''
    dfpl = df[:].copy()

    def SIGNAL():
        return dfpl.ordersignal

    class MyStrat(Strategy):
        mysize = 0.05 #1000
        def init(self):
            super().init()
            self.signal = self.I(SIGNAL)

        def next(self):
            super().next()
            TPSLRatio = 2
            perc = 0.02 

            # Here is the difference between worse and better
            # NOTE: important difference here
            # since we always do two trades: tp1 and tp2
            # if one has been closed, we should adapt the sl of the second (tp1)
            # to a new value, which is:
            # changing the stoploss to the entry price of the last trade?
            if len(self.trades)==1:
                self.trades[-1].sl = self.trades[-1].entry_price

            if self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==2:
                sl1 = self.data.Close[-1]-self.data.Close[-1]*perc
                sldiff = abs(sl1-self.data.Close[-1])
                tp1 = self.data.Close[-1]+sldiff*TPSLRatio
                tp2 = self.data.Close[-1]+sldiff
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
                self.buy(sl=sl1, tp=tp2, size=self.mysize)

            elif self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==1:
                sl1 = self.data.Close[-1]+self.data.Close[-1]*perc
                sldiff = abs(sl1-self.data.Close[-1])
                tp1 = self.data.Close[-1]-sldiff*TPSLRatio
                tp2 = self.data.Close[-1]-sldiff
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
    
    # margin controls the leverage
    # 1/100 is a higher leverage than 1/10
    bt = Backtest(dfpl, MyStrat, cash=10000, margin=1/100, commission=.000)
    stat = bt.run()
    bt.plot(show_legend=False)
    print(stat._trades)
    print(stat)
    return stat




if __name__ == '__main__':
    dfSPY = read_data()
    dfSPY = add_ema_signal(dfSPY, 3)
    dfSPY = add_orders_limit(dfSPY, 0.01)
    dfSPY = add_pointpos(dfSPY)
    # plot_signal(dfSPY)   # will observe a few buy signals
    # stat = backtest_simple(dfSPY)
    # stat = backtest_worse(dfSPY)
    stat = backtest_better(dfSPY)
