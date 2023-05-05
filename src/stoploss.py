"""
Four types of Stop Loss
1. Fixed difference
2. ATR difference: ATR measures the volativity
3. Fixed Trail
4. ATR Trail
"""



import numpy as np
import pandas as pd
import pandas_ta as pa
from backtesting import Strategy, Backtest






def read_data():
    df = pd.read_csv("EURUSD_Candlestick_1_D_ASK_05.05.2003-30.06.2021.csv")
    df.columns = ['Local time', 'open', 'high', 'low', 'close', 'volume']
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.isna().sum()
    df.reset_index(drop=True, inplace=True)
    df['ATR'] = pa.atr(high=df.high, low=df.low, close=df.close, length=14)
    return df



# read martingale_swing.py for details
def support(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.low[i]>df1.low[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.low[i]<df1.low[i-1]):
            return 0
    return 1



# read martingale_swing.py for details
def resistance(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.high[i]<df1.high[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.high[i]>df1.high[i-1]):
            return 0
    return 1



# read martingale_swing.py for details
def add_signal(df, n1=2, n2=2, backCandles=45):
    length = len(df)
    high = list(df['high'])
    low = list(df['low'])
    close = list(df['close'])
    open = list(df['open'])
    bodydiff = [0] * length

    highdiff = [0] * length
    lowdiff = [0] * length
    ratio1 = [0] * length
    ratio2 = [0] * length

    # EurUSD set
    mybodydiff = 0.000001
    mybodydiffmin = 0.002

    def isEngulfing(l):
        row=l
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<mybodydiff:
            bodydiff[row]=mybodydiff

        bodydiffmin = mybodydiffmin
        if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            open[row-1]<close[row-1] and
            open[row]>close[row] and
            (open[row]-close[row-1])>=-0e-5 and close[row]<open[row-1]): #+0e-5 -5e-5
            return 1

        elif(bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            open[row-1]>close[row-1] and
            open[row]<close[row] and
            (open[row]-close[row-1])<=+0e-5 and close[row]>open[row-1]):#-0e-5 +5e-5
            return 2
        else:
            return 0

    def isStar(l):
        bodydiffmin = mybodydiffmin
        row=l
        highdiff[row] = high[row]-max(open[row],close[row])
        lowdiff[row] = min(open[row],close[row])-low[row]
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<mybodydiff:
            bodydiff[row]=mybodydiff
        ratio1[row] = highdiff[row]/bodydiff[row]
        ratio2[row] = lowdiff[row]/bodydiff[row]

        if (ratio1[row]>1 and lowdiff[row]<0.2*highdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]>close[row]):
            return 1
        elif (ratio2[row]>1 and highdiff[row]<0.2*lowdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]<close[row]):
            return 2
        else:
            return 0

    def closeResistance(l,levels,lim):
        if len(levels)==0:
            return 0
        #!!!
        #lim=df.ATR[l]/2

        #diff between high and closest level among levels
        c1 = abs(df.high[l]-min(levels, key=lambda x:abs(x-df.high[l])))<=lim
        #diff between higher body and closest level to high
        c2 = abs(max(df.open[l],df.close[l])-min(levels, key=lambda x:abs(x-df.high[l])))<=lim
        #min body less than closest level to high
        c3 = min(df.open[l],df.close[l])<min(levels, key=lambda x:abs(x-df.high[l]))
        #low price less than closest level to high
        c4 = df.low[l]<min(levels, key=lambda x:abs(x-df.high[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1
        else:
            return 0

    def closeSupport(l,levels,lim):
        if len(levels)==0:
            return 0
        #!!!
        #lim=df.ATR[l]/2
        c1 = abs(df.low[l]-min(levels, key=lambda x:abs(x-df.low[l])))<=lim
        c2 = abs(min(df.open[l],df.close[l])-min(levels, key=lambda x:abs(x-df.low[l])))<=lim
        c3 = max(df.open[l],df.close[l])>min(levels, key=lambda x:abs(x-df.low[l]))
        c4 = df.high[l]>min(levels, key=lambda x:abs(x-df.low[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1
        else:
            return 0

    signal = [0] * length

    for row in range(backCandles, len(df)-n2):
        ss = []
        rr = []
        for subrow in range(row-backCandles+n1, row+1):
            if support(df, subrow, n1, n2):
                ss.append(df.low[subrow])
            if resistance(df, subrow, n1, n2):
                rr.append(df.high[subrow])

        #!!!! parameters
        myclosedistance = 150e-5 #EURUSD
        if ((isEngulfing(row)==1 or isStar(row)==1) and closeResistance(row, rr, myclosedistance) ):#and df.RSI[row]<30
            signal[row] = 1
        elif((isEngulfing(row)==2 or isStar(row)==2) and closeSupport(row, ss, myclosedistance)):#and df.RSI[row]>70
            signal[row] = 2
        else:
            signal[row] = 0

    df['signal']=signal
    df.columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'signal']

    return df



def backtest1(df):
    """
    fixed difference stoploss
    """
    def SIGNAL():
        return df.signal
    class MyCandlesStrat(Strategy):
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            # bullish
            if self.signal1==2:
                sl1 = self.data.Close[-1] - 550e-4 #EURUSD
                tp1 = self.data.Close[-1] + 600e-4
                self.buy(sl=sl1, tp=tp1)
            # bearish
            elif self.signal1==1:
                sl1 = self.data.Close[-1] + 550e-4 #EURUSD
                tp1 = self.data.Close[-1] - 600e-4
                self.sell(sl=sl1, tp=tp1)

    bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.000)
    stat = bt.run()
    bt.plot()
    print(stat, "\n")
    return stat



def backtest2(df):
    """
    ATR difference stoploss
    """
    def SIGNAL():
        return df.signal
    class MyCandlesStrat(Strategy):
        atr_f = 0.2
        ratio_f = 1
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            # bullish
            if self.signal1==2:
                sl1 = self.data.Close[-1] - self.data.ATR[-1]/self.atr_f
                tp1 = self.data.Close[-1] + self.data.ATR[-1]*self.ratio_f/self.atr_f
                self.buy(sl=sl1, tp=tp1)
            # bearish
            elif self.signal1==1:
                sl1 = self.data.Close[-1] + self.data.ATR[-1]/self.atr_f
                tp1 = self.data.Close[-1] - self.data.ATR[-1]*self.ratio_f/self.atr_f
                self.sell(sl=sl1, tp=tp1)

    bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.000)
    stat = bt.run()
    bt.plot()
    print(stat, "\n")
    return stat



def backtest3(df):
    """
    fixed trail stoploss
    """
    def SIGNAL():
        return df.signal
    class MyCandlesStrat(Strategy):
        # the trailing value
        sltr=500e-4 #EURUSD
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            sltr = self.sltr
            # when there are some trades
            for trade in self.trades:
                # if the trade is long
                # set the stoploss of the trade to be the 
                # max(current close - tral value, predef sl of this trade)
                if trade.is_long:
                    trade.sl = \
                        max(trade.sl or -np.inf, self.data.Close[-1] - sltr)
                # if the trade is short
                # set the stoploss of the trade to be the 
                # min(current close + tral value, predef sl of this trade)
                else:
                    trade.sl = \
                        min(trade.sl or np.inf, self.data.Close[-1] + sltr)
            # when there is no trade yet
            if self.signal1==2 and len(self.trades)==0: # trades number change!
                sl1 = self.data.Close[-1] - sltr
                self.buy(sl=sl1)
            elif self.signal1==1 and len(self.trades)==0: # trades number change!
                sl1 = self.data.Close[-1] + sltr
                self.sell(sl=sl1)


    bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.000)
    stat = bt.run()
    bt.plot()
    print(stat, "\n")
    return stat



def backtest4(df):
    """
    ATR trail stoploss
    """
    def SIGNAL():
        return df.signal
    class MyCandlesStrat(Strategy):
        atr_f = 1
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            for trade in self.trades:
                if trade.is_long:
                    trade.sl = max(trade.sl or -np.inf, self.data.Close[-1] - self.data.ATR[-1]/self.atr_f)
                else:
                    trade.sl = min(trade.sl or np.inf, self.data.Close[-1] + self.data.ATR[-1]/self.atr_f)

            if self.signal1==2 and len(self.trades)==0: # trades number change!
                sl1 = self.data.Close[-1] - self.data.ATR[-1]/self.atr_f
                self.buy(sl=sl1)
            elif self.signal1==1 and len(self.trades)==0: # trades number change!
                sl1 = self.data.Close[-1] + self.data.ATR[-1]/self.atr_f
                self.sell(sl=sl1)
    bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.000)
    stat = bt.run()
    bt.plot()
    print(stat, "\n")
    return stat







if __name__ == "__main__":
    df = read_data()
    n1=2
    n2=2
    backCandles=45
    df = add_signal(df, n1, n2, backCandles)

    stat1 = backtest1(df)
    stat2 = backtest2(df)
    stat3 = backtest3(df)
    stat4 = backtest4(df)

