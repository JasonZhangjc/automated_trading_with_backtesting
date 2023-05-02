"""
Martingale approach with Swing strategy
    - Resistance and Support
"""




import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting import Backtest







def read_data():
    df = pd.read_csv("EURUSD_Candlestick_1_D_ASK_05.05.2003-30.06.2021.csv")
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df.isna().sum()
    return df



def support(df1, curr, n1, n2): #n1 n2 before and after candle l
    """
    support returns 1 if curr low price is a local miminum
    """
    for i in range(curr-n1+1, curr+1):
        if(df1.low[i]>df1.low[i-1]):
            return 0
    for i in range(curr+1,curr+n2+1):
        if(df1.low[i]<df1.low[i-1]):
            return 0
    return 1



def resistance(df1, curr, n1, n2): #n1 n2 before and after candle l
    """
    resistance returns 1 if curr high price is a local maximum
    """
    for i in range(curr-n1+1, curr+1):
        if(df1.high[i]<df1.high[i-1]):
            return 0
    for i in range(curr+1,curr+n2+1):
        if(df1.high[i]>df1.high[i-1]):
            return 0
    return 1



def add_signal(df, n1, n2, back_candles):
    """
    This function is complicated
    """
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

    def isEngulfing(l):
        row=l
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<0.000001:
            bodydiff[row]=0.000001
        # bodydiffmin is to avoid too small candles
        bodydiffmin = 0.002
        # if r is a bearish engulfing candle
        if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            open[row-1]<close[row-1] and
            open[row]>close[row] and
            (open[row]-close[row-1])>=-0e-5 and close[row]<open[row-1]): #+0e-5 -5e-5
            return 1
        # if r is a bullish engulfing candl
        elif(bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            open[row-1]>close[row-1] and
            open[row]<close[row] and
            (open[row]-close[row-1])<=+0e-5 and close[row]>open[row-1]):#-0e-5 +5e-5
            return 2
        else:
            return 0

    def isStar(l):
        bodydiffmin = 0.0020
        row=l
        highdiff[row] = high[row]-max(open[row],close[row])
        lowdiff[row] = min(open[row],close[row])-low[row]
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<0.000001:
            bodydiff[row]=0.000001
        ratio1[row] = highdiff[row]/bodydiff[row]
        ratio2[row] = lowdiff[row]/bodydiff[row]
        # if highdiff is larger than the body diff
        # if lowdiff is small compared to highdiff
        # if bodydiff is not too small
        # long long neck, short short leg, like a reverse hammer
        if (ratio1[row]>1 and 
            lowdiff[row]<0.2*highdiff[row] and 
            bodydiff[row]>bodydiffmin):# and open[row]>close[row]):
            return 1
        # if lowdiff is larger than the body diff
        # if highdiff is small compared to lowdiff
        # if bodydiff is not too small
        # short short neck, long long leg, like a hammer
        elif (ratio2[row]>1 and 
              highdiff[row]<0.2*lowdiff[row] and 
              bodydiff[row]>bodydiffmin):# and open[row]<close[row]):
            return 2
        else:
            return 0

    def close_to_resistance(l,levels,lim):
        if len(levels)==0:
            return 0
        # evaluate 4 conditions
        c1 = abs(df.high[l] - 
                 min(levels, key=lambda x:abs(x-df.high[l]))) <= lim
        c2 = abs(max(df.open[l],df.close[l]) - 
                 min(levels, key=lambda x:abs(x-df.high[l]))) <= lim
        c3 = min(df.open[l],df.close[l]) < min(levels, key=lambda x:abs(x-df.high[l]))
        c4 = df.low[l] < min(levels, key=lambda x:abs(x-df.high[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1
        else:
            return 0

    def close_to_support(l,levels,lim):
        if len(levels)==0:
            return 0
        # evaluate 4 conditions
        c1 = abs(df.low[l] - 
                 min(levels, key=lambda x:abs(x-df.low[l]))) <= lim
        c2 = abs(min(df.open[l],df.close[l]) - 
                 min(levels, key=lambda x:abs(x-df.low[l]))) <= lim
        c3 = max(df.open[l],df.close[l]) > min(levels, key=lambda x:abs(x-df.low[l]))
        c4 = df.high[l] > min(levels, key=lambda x:abs(x-df.low[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1
        else:
            return 0

    # main loop
    signal = [0] * length
    for row in range(back_candles, len(df)-n2):
        ss = []
        rr = []
        for subrow in range(row-back_candles+n1, row+1):
            # if the subrow candle is a local min
            if support(df, subrow, n1, n2):
                ss.append(df.low[subrow])
            # if the subrow candle is a local opt
            if resistance(df, subrow, n1, n2):
                rr.append(df.high[subrow])
        # if Engulfing and star indicates bearish
        # if the candle is close to resistance
        # output bearish
        if ((isEngulfing(row)==1 or isStar(row)==1) and 
            close_to_resistance(row, rr, 150e-5) ):#and df.RSI[row]<30
            signal[row] = 1
        # if Engulfing and star indicates bullish
        # if the candle is close to support
        # output bullish
        elif((isEngulfing(row)==2 or isStar(row)==2) and 
             close_to_support(row, ss, 150e-5)):#and df.RSI[row]>70
            signal[row] = 2
        else:
            signal[row] = 0

    df['signal']=signal
    df.columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal']

    return df



def backtest(df):
    """
    We use leverage here by setting leverage=1/10 in Backtest
    However, we need to make sure that we have a high winning rate via backtest,
    such that we can get more return with leverage. Otherwise, low winning rate with high leverage will enlarge the risk and ruin our trading.
    """
    def SIGNAL():
        return df.signal

    class MyCandlesStrat(Strategy):
        # initsize is the amount of money we use for each trade initially
        initsize = 0.05
        mysize = initsize
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            if(self.signal1>0 and len(self.trades)==0 and len(self.closed_trades)>0 and self.closed_trades[-1].pl < 0):
                self.mysize=self.mysize*2
            elif len(self.closed_trades)>0 and self.closed_trades[-1].pl > 0:
                self.mysize=self.initsize
            if self.signal1==2:
                sl1 = self.data.Close[-1] - 450e-4
                tp1 = self.data.Close[-1] + 450e-4
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
            elif self.signal1==1:
                sl1 = self.data.Close[-1] + 450e-4
                tp1 = self.data.Close[-1] - 450e-4
                self.sell(sl=sl1, tp=tp1, size=self.mysize)
    # <margin> parameter is the leverage!
    bt = Backtest(df, MyCandlesStrat, cash=10000, margin=1/10, commission=.00)
    stat = bt.run()
    bt.plot(show_legend=False)
    print(stat)
    return stat






if __name__ == "__main__":
    df = read_data()
    n1, n2, back_candles = 2, 2, 30
    df = add_signal(df, n1, n2, back_candles)
    backtest(df)

