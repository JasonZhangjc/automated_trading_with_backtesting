"""
Grid trading strategy
"""




import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Strategy
from backtesting import Backtest
import backtesting




def read_data_eurousd():
    df = yf.download("EURUSD=X", start="2023-03-11", end="2023-04-30", interval='5m')
    return df


def generate_grid(mid_price, grid_distance, grid_range):
    return (np.arange(mid_price-grid_range, mid_price+grid_range, grid_distance))


def add_signal(df):
    signal = [0]*len(df)
    i=0
    for index, row in df.iterrows():
        for p in grid:
            if min(row.Low, row.High)<p and max(row.Low, row.High)>p:
                signal[i]=1
        i+=1
    df["signal"]=signal
    df[df["signal"]==1]


def grid_backtest(df):
    """
    Does not perform well
    """
    dfpl = df[:].copy()
    dfpl['ATR'] = ta.atr(high = dfpl.High, low = dfpl.Low, close = dfpl.Close, length = 16)
    dfpl.dropna(inplace=True)

    def SIGNAL():
        return dfpl.signal

    class MyStrat(Strategy):
        mysize = 50
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            slatr = 1.5*grid_distance #5*self.data.ATR[-1]
            TPSLRatio = 0.5

            if self.signal1==1 and len(self.trades)<=10000:
                sl1 = self.data.Close[-1] + slatr
                tp1 = self.data.Close[-1] - slatr*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

                sl1 = self.data.Close[-1] - slatr
                tp1 = self.data.Close[-1] + slatr*TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)

    # set hedging to True to allow buy and sell at the same time
    bt = Backtest(dfpl, MyStrat, cash=1000, margin=1/10, hedging=True, exclusive_orders=False)
    stat = bt.run()

    backtesting.set_bokeh_output(notebook=False)
    bt.plot(show_legend=False, plot_width=None, plot_equity=True, plot_return=False, 
    plot_pl=False, plot_volume=False, plot_drawdown=False, smooth_equity=False, relative_equity=True, 
    superimpose=True, resample=False, reverse_indicators=False, open_browser=True)

    return stat





if __name__ == "__main__":
    df = read_data_eurousd()
    grid_distance = 0.01
    mid_price = 1.065
    grid_range=0.1
    grid = generate_grid(mid_price=mid_price, grid_distance=grid_distance, grid_range=0.1)
    add_signal(df)
    # print(df)

    stat = grid_backtest(df)
    # print(stat._trades.sort_values(by="EntryBar").head(20))

    print(stat)