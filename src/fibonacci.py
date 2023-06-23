"""
Fibonacci Retracement Strategy
"""




import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from backtesting import Strategy
from backtesting import Backtest




def read_data():
    '''
    read data from .csv file and generate RSI and EMA indicators
    '''
    df = pd.read_csv("../data/EURUSD_Candlestick_1_Hour_BID_04.05.2003-15.04.2023.csv")
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df.isna().sum()
    # generate RSI and EMA indicators
    df['RSI'] = ta.rsi(df.close, length=12)
    df['EMA'] = ta.ema(df.close, length=150)
    df=df[0:2000]
    
    return df



def generate_ema_signal(df, backcandles=15):
    '''
    generate EMA signal
    moving average
    '''
    EMAsignal = [0]*len(df)

    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if max(df.open[i], df.close[i])>=df.EMA[i]:
                dnt=0
            if min(df.open[i], df.close[i])<=df.EMA[i]:
                upt=0
        if upt==1 and dnt==1:
            EMAsignal[row]=3    # not clear trend
        elif upt==1:
            EMAsignal[row]=2    # uptrend
        elif dnt==1:
            EMAsignal[row]=1    # downtrend

    df['EMASignal'] = EMAsignal

    return df



def generate_fibonacci_signal(df, l, backcandles, gap, 
    zone_threshold, price_diff_threshold):
    '''
    Generate Fibonacci signals
    return:
        downtrend = 1, uptrend = 2
        Take profit
        Stop loss
        min idx 
        max idx
    '''
    max_price = df.high[l-backcandles:l-gap].max()
    min_price = df.low[l-backcandles:l-gap].min()
    index_max = df.high[l-backcandles:l-gap].idxmax()
    index_min = df.low[l-backcandles:l-gap].idxmin()
    price_diff = max_price - min_price

    if (df.EMASignal[l] == 2           # if uptrend
        and (index_min < index_max)    # if min idx is earlier than mx idx
        and price_diff>price_diff_threshold):    # if price difference is nontrivial
        l1 = max_price - 0.62 * price_diff # position entry 0.62
        l2 = max_price - 0.78 * price_diff # SL 0.78
        l3 = max_price - 0. * price_diff # TP
        if (abs(df.close[l]-l1) < zone_threshold and # if golden zone is not too small
            df.high[l-gap:l].min()>l1):    # if the values in the gap > the golden zone
            return (2, l2, l3, index_min, index_max)
        else:
            return (0,0,0,0,0)

    elif (df.EMASignal[l] == 1         # if downtrend
          and (index_min > index_max)  # if min idx is later than max idx
          and price_diff>price_diff_threshold):  # if price difference is not trivial
        l1 = min_price + 0.62 * price_diff # position entry 0.62
        l2 = min_price + 0.78 * price_diff # SL 0.78
        l3 = min_price + 0. * price_diff # TP
        if (abs(df.close[l]-l1) < zone_threshold # if golden zone is not too small
            and df.low[l-gap:l].max()<l1):  # if the values in the gap < the golden zone    
            return (1, l2, l3, index_min, index_max)
        else:
            return (0,0,0,0,0)

    else:
        return (0,0,0,0,0)



def generate_signal(df, gap_candles = 5, backcandles = 40,
                    zone_threshold=0.001, 
                    price_diff_threshold=0.01):
    '''
    generate the Fibonacci signals and put them in the df
    '''
    signal = [0 for i in range(len(df))]
    TP = [0 for i in range(len(df))]
    SL = [0 for i in range(len(df))]
    MinSwing = [0 for i in range(len(df))]
    MaxSwing = [0 for i in range(len(df))]

    for row in range(backcandles, len(df)):
        gen_sig = generate_fibonacci_signal(df, row, backcandles=backcandles, 
            gap=gap_candles, zone_threshold=zone_threshold, 
            price_diff_threshold=price_diff_threshold)
        signal[row] = gen_sig[0]
        SL[row] = gen_sig[1]
        TP[row] = gen_sig[2]
        MinSwing[row] = gen_sig[3]
        MaxSwing[row] = gen_sig[4]

    df['signal'] = signal
    df['SL'] = SL
    df['TP'] = TP
    df['MinSwing'] = MinSwing
    df['MaxSwing'] = MaxSwing

    return df



def add_pointpos(df):
    def pointpos(x):
        if x['signal']==1:
            return x['high']+1e-4
        elif x['signal']==2:
            return x['low']-1e-4
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    
    return df



def plot_signal(df):
    '''
    Plot the signals
    '''
    dfpl = df[150:350]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['open'],
                    high=dfpl['high'],
                    low=dfpl['low'],
                    close=dfpl['close'])])

    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
        paper_bgcolor='black',
        plot_bgcolor='black')
    fig.update_xaxes(gridcolor='black')
    fig.update_yaxes(gridcolor='black')
    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=8, color="MediumPurple"),
                    name="Signal")
    fig.show()



def backtest(df):
    '''
    backtesting with our generated signals
    '''
    df = df.rename(columns={"open": "Open", "high":"High", 
                            "low":"Low", "close": "Close", "volume":"Volume"})
    def SIGNAL():
        return df.signal

    class MyStrat(Strategy):
        mysize = 0.99 #1000
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()

            if self.signal1==2 and len(self.trades)==0:
                sl1 = self.data.SL[-1]
                tp1 = self.data.TP[-1]
                tp2 = tp1-(tp1-self.data.Close[-1])/2
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
                self.buy(sl=sl1, tp=tp2, size=self.mysize)

            elif self.signal1==1 and len(self.trades)==0:
                sl1 = self.data.SL[-1]
                tp1 = self.data.TP[-1]
                tp2 = tp1+(self.data.Close[-1]-tp1)/2
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(df, MyStrat, cash=100, margin=1/100, commission=0.0000)
    stat = bt.run()
    print(stat)
    bt.plot()

    return stat






if __name__ == '__main__':
    df = read_data()
    backcandles=15
    df = generate_ema_signal(df, backcandles)
    df = generate_signal(df, gap_candles = 5, backcandles = 40,
        zone_threshold = 0.001, price_diff_threshold = 0.01)
    df = add_pointpos(df)
    plot_signal(df)
    backtest(df)

