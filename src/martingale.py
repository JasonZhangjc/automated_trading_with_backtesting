"""
Martingale Approach
    - Double the bet in the same direction if after a loss
    - Will win all money back once the other direction appears
    - However, this is a little bit ideal
    - How bets ruin lives
Simple backtest
"""



import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting import Backtest





def read_data():
    df = pd.read_csv("EURUSD_Candlestick_1_D_ASK_05.05.2003-30.06.2021.csv")
    #df = pd.read_csv("EURUSD_Candlestick_1_Hour_ASK_05.05.2003-15.01.2022.csv")
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df.isna().sum()
    return df




def add_simple_signal(df):
    df['signal'] = np.random.random(len(df))
    # add random signal and make them 1 or 2 to indicate down or up
    df['signal'] = df['signal'].apply(lambda x: 1 if x<0.5 else 2)
    # df[df['signal']==1].count()+df[df['signal']==2].count()
    df.columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal']
    return df



def backtest(df):
    def SIGNAL():
        return df.signal

    class MyStrat(Strategy):
        initsize = 1
        mysize = initsize
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            # if the sinal > 0, it means we have either buy or sell
            # len(self.trades)==0 means we don't have open trade
            # len(self.closed_trades)>0 means we have closed some trades
            # pl: profit loss < 0 means we lose money
            # if we lose money we double the bet
            # we apply the Martingale
            if (self.signal1>0 and
                len(self.trades)==0 and
                len(self.closed_trades)>0 and
                self.closed_trades[-1].pl < 0):
                self.mysize=self.mysize*2
                #print(self.data.index, self.mysize)
                #print(self.closed_trades[-1], self.closed_trades[-1].pl)
                #print(self.equity)
                #print("-"*20)
            # if we have closed some trades
            # if pl > 0, it means we earn some money,
            # we use the initial size
            # we don't apply Martingale
            elif (len(self.closed_trades)>0 and
                  self.closed_trades[-1].pl > 0):
                self.mysize=self.initsize

            # if we don't have open trades,
            # if the current signal is bullish, we buy
            if self.signal1==2 and len(self.trades)==0:
                sl1 = self.data.Close[-1] - 600e-4
                tp1 = self.data.Close[-1] + 600e-4
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
            # if we don't have open trades,
            # if the current signal is bearish, we sell
            elif self.signal1==1 and len(self.trades)==0:
                sl1 = self.data.Close[-1] + 600e-4
                tp1 = self.data.Close[-1] - 600e-4
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
    bt = Backtest(df, MyStrat, cash=10000, commission=.00)
    stat = bt.run()
    bt.plot(show_legend=False)
    print(stat)
    return stat




if __name__ == "__main__":
    df = read_data()
    df = add_simple_signal(df)
    backtest(df)