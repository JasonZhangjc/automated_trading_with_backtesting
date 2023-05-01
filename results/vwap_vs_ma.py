"""
Strategy Overview:
- use MA or VWAP as reference
- if we find a region below the ref, it means the security is in a descending trend, then we should consider a short position. The next question is where is our short time point.
- the security price would re-touch the ref after a long time below, thus, the touching point is our time point.
- similarly for the long position

use ATR for backtesting
"""



import pandas as pd
import pandas_ta as ta    # technical analysis
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest




def read_data_bitcoin():
    """
    Read bitcoin data
    """
    df = pd.read_csv("BTCUSD_Candlestick_15_M_ASK_05.08.2019-29.04.2022.csv")

    return df



def format_gmt_time(df):
    """
    format GMT time 
    """
    df["Gmt time"]=df["Gmt time"].str.replace(".000","")
    df['Gmt time']=pd.to_datetime(df['Gmt time'],format='%d.%m.%Y %H:%M:%S')
    df.set_index("Gmt time", inplace=True)
    df=df[df.High!=df.Low]

    return df



def add_vwap_and_ema(df):
    """
    add VWAP and EMA data computed via technical analysis
    """
    # volume weighted average price
    df["VWAP"]=ta.vwap(df.High, df.Low, df.Close, df.Volume)
    # estimated moving average
    df["EMA"]=ta.ema(df.Close, length=100)

    return df



def signal_ema(df, back_candles):
    """
    add ema signal to df
    """
    ema_signal = [0]*len(df)

    for row in tqdm(range(back_candles, len(df))):
        upt = 1
        dnt = 1
        for i in range(row - back_candles, row + 1):
            if df.High[i] >= df.EMA[i]:
                dnt=0
            if df.Low[i] <= df.EMA[i]:
                upt=0
        # 1: short signal
        # 2: long signal
        # 0 or 3: no particular signal
        if upt==1 and dnt==1:
            ema_signal[row]=3
        elif upt==1:
            ema_signal[row]=2
        elif dnt==1:
            ema_signal[row]=1

    df['EMASignal'] = ema_signal



def signal_vwap(df, back_candles):
    """
    add vwap signal to df
    """
    vwap_signal = [0]*len(df)
    back_candles = 6

    for row in tqdm(range(back_candles, len(df))):
        upt = 1
        dnt = 1
        for i in range(row - back_candles, row + 1):
            if df.High[i] >= df.VWAP[i]:
                dnt=0
            if df.Low[i] <= df.VWAP[i]:
                upt=0
        # 1: short signal
        # 2: long signal
        # 0 or 3: no particular signal
        if upt==1 and dnt==1:
            vwap_signal[row]=3
        elif upt==1:
            vwap_signal[row]=2
        elif dnt==1:
            vwap_signal[row]=1

    df['VWAPSignal'] = vwap_signal



def total_ema_signal(df, row, my_close_distance):
    """
    calculate the total ema signal
    """
    # long signal: all back_candles are above ema
    if (df.EMASignal[row]==2
        # look for long entry point, if found, long
        and min(abs(df.EMA[row]-df.High[row]),
                abs(df.EMA[row]-df.Low[row])) <= my_close_distance):
            return 2
    # short signal: all back_candles are below ema
    if (df.EMASignal[row]==1
        # look for short entry point, if found, short
        and min(abs(df.EMA[row]-df.High[row]),
                abs(df.EMA[row]-df.Low[row])) <= my_close_distance):
            return 1



def total_vwap_signal(df, row, my_close_distance):
    """
    calculate the total vwap signal
    """
    # long signal: all back_candles are above vwap
    if (df.VWAPSignal[row]==2
        # look for long entry point, if found, long
        and min(abs(df.VWAP[row]-df.High[row]),
                abs(df.VWAP[row]-df.Low[row])) <= my_close_distance):
            return 2
    # short signal: all back_candles are below vwap
    if (df.VWAPSignal[row]==1
        # look for short entry point, if found, short
        and min(abs(df.VWAP[row]-df.High[row]),
                abs(df.VWAP[row]-df.Low[row])) <= my_close_distance):
            return 1



def total_signal(df, my_close_distance, ref='vwap'):
     # Change signal Here VWAP and EMA
    tot_signal = [0]*len(df)
    if ref == 'vwap' or 'VWAP':
        for row in range(0, len(df)): #careful backcandles used previous cell
            tot_signal[row] = total_vwap_signal(df, row, my_close_distance)
    else:
         for row in range(0, len(df)): #careful backcandles used previous cell
            tot_signal[row] = total_ema_signal(df, row, my_close_distance)
    df['TotalSignal'] = tot_signal



def point_pos_break(df):
    if df['TotalSignal'] == 1:
        return df['High'] + 1e-3
    elif df['TotalSignal'] == 2:
        return df['Low'] - 1e-3
    else:
        return np.nan



def add_point_pos_break(df):
    """
    add point_pos_break
    """
    df['pointposbreak'] = \
        df.apply(lambda row: point_pos_break(row), axis=1)



def plot_ema_and_vwap(df):
    """
    plot the df by using plotly
    """
    dfpl = df[150:400]
    dfpl.reset_index(inplace=True)
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=1), name="EMA"),
                    go.Scatter(x=dfpl.index, y=dfpl.VWAP, line=dict(color='blue', width=1), name="VWAP")])
    fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="Signal")
    fig.show()



def add_ATR(dfpl):
    """
    add a column for ATR in dfpl
    """
    dfpl['ATR']=ta.atr(dfpl.High, dfpl.Low, dfpl.Close, length=5)



def backtest_strategy_ATR(dfpl):
    """
    backtesting the strategy with ATR
    This model works well when we have large magnitude movement or volativity

    we can tune parameters to improve the performance
    e.g., if we want to increase the Sharpe ratio, we can decrease the margin
    """
    def SIGNAL():
        return dfpl.TotalSignal

    class MyStrat(Strategy):
        initsize = 0.5
        mysize = initsize

        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            # stop loss = 0.8
            # stop loss position is ATR related
            sl_atr = 0.8 * self.data.ATR[-1]
            # take profit stop loss ratio = 2.5
            # the take profit position == 2.5 * stop loss position
            TPSL_Ratio = 2.5

            # long position
            # we allow one trade at a time so len(self.trades)==0
            if self.signal1==2 and len(self.trades)==0:
                # if the price goes below sl1, we sell but bear a loss to stop further potential loss in the future
                sl1 = self.data.Close[-1] - sl_atr
                # if the price goes above tp1, we buy and bear a potential higher price in the future
                tp1 = self.data.Close[-1] + sl_atr*TPSL_Ratio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)

            # short position
            # we allow one trade at a time so len(self.trades)==0
            elif self.signal1==1 and len(self.trades)==0:
                # if the price goes above sl1, we sell but bear a loss to stop further potential loss in the future
                sl1 = self.data.Close[-1] + sl_atr
                # if the price goes below tp1, we buy and bear a potential lower price in the future
                tp1 = self.data.Close[-1] - sl_atr*TPSL_Ratio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    # the margin can be tuned
    bt = Backtest(dfpl, MyStrat, cash=100000, margin=1/5, commission=0.0000)
    stat = bt.run()
    bt.plot()

    return stat








if __name__ == "__main__":
    df = read_data_bitcoin()
    df = format_gmt_time(df)

    df = add_vwap_and_ema(df)

    back_candles = 6
    signal_ema(df, back_candles)
    signal_vwap(df, back_candles)

    my_close_distance = 100
    ref = 'vwap'
    total_signal(df, my_close_distance, ref)

    add_point_pos_break(df)
    plot_ema_and_vwap(df)

    dfpl = df[:].copy()
    add_ATR(dfpl)
    stat = backtest_strategy_ATR(dfpl)
    print(stat)