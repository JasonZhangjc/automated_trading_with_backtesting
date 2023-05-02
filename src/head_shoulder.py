"""
Detect head and shoulder patterns with linear regression
"""



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import linregress






def read_data():
    # df = pd.read_csv("EURUSD_Candlestick_4_Hour_ASK_05.05.2003-16.10.2021.csv")
    df = pd.read_csv("EURUSD_Candlestick_1_Hour_BID_05.05.2003-08.02.2022.csv")
    df.columns=['time', 'open', 'high', 'low', 'close', 'volume']
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df.isna().sum()
    return df



def add_pivot(df):
    """
    add pivotLow and pivotHigh
    """
    def pivotid(df1, l, n1, n2): #n1 n2 before and after candle l
        if l-n1 < 0 or l+n2 >= len(df1):
            return 0
        pividlow=1
        pividhigh=1
        for i in range(l-n1, l+n2+1):
            if(df1.low[l]>df1.low[i]):
                pividlow=0
            if(df1.high[l]<df1.high[i]):
                pividhigh=0
        if pividlow and pividhigh:
            return 3
        elif pividlow:
            return 1
        elif pividhigh:
            return 2
        else:
            return 0
    # pivot is long pivot, shortpivot is short pivot
    df['pivot'] = df.apply(lambda x: pivotid(df, x.name,15,15), axis=1)
    df['shortpivot'] = df.apply(lambda x: pivotid(df, x.name,5,5), axis=1)
    return df



def add_point_pos(df):
    def point_pos(x):
        if x['pivot']==1:
            return x['low']-1e-3
        elif x['pivot']==2:
            return x['high']+1e-3
        else:
            return np.nan
    def short_point_pos(x):
        if x['shortpivot']==1:
            return x['low']-2e-3
        elif x['shortpivot']==2:
            return x['high']+2e-3
        else:
            return np.nan
    df['pointpos'] = df.apply(lambda row: point_pos(row), axis=1)
    df['shortpointpos'] = df.apply(lambda row: short_point_pos(row), axis=1)
    return df



def plot_signal(df, start, end):
    dfpl = df[start:end]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['open'],
                    high=dfpl['high'],
                    low=dfpl['low'],
                    close=dfpl['close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")
    fig.add_scatter(x=dfpl.index, y=dfpl['shortpointpos'], mode="markers",
                    marker=dict(size=5, color="red"),
                    name="shortpivot")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()



def detect_head_and_shoulders(df, back_candles):
    """
    only for the up direction
        - leftmost = -back_candles
        - rightmost = back_candles
    """
    for candleid in range(8000, len(df)-back_candles):
        if df.iloc[candleid].pivot != 2 or df.iloc[candleid].shortpivot != 2:
            continue

        # if candle is both a pivot and a short pivot
        # it might be a head
        maxim = np.array([])   # y coordinates for local minimum
        minim = np.array([])   # y coordinates for local maximum
        xxmin = np.array([])   # x coordinates for local minimum
        xxmax = np.array([])   # x coordinates for local maximum
        minbcount=0 #minimas before head
        maxbcount=0 #maximas before head
        minacount=0 #minimas after head
        maxacount=0 #maximas after head

        # [-back_candles, back_candles]
        # left back_candles
        # right back_candles
        for i in range(candleid-back_candles, candleid+back_candles):
            if df.iloc[i].shortpivot == 1:
                minim = np.append(minim, df.iloc[i].low)
                xxmin = np.append(xxmin, i)
                #could be i instead df.iloc[i].name
                if i < candleid:
                    minbcount=+1
                elif i>candleid:
                    minacount+=1
            if df.iloc[i].shortpivot == 2:
                # the candleid will also be included in maxim and xxmax
                maxim = np.append(maxim, df.iloc[i].high)
                xxmax = np.append(xxmax, i) # df.iloc[i].name
                if i < candleid:
                    maxbcount+=1
                elif i>candleid:
                    maxacount+=1
        # if we don't have local minimum or local maximum, 
        # we do not have head and shoulders 
        if minbcount<1 or minacount<1 or maxbcount<1 or maxacount<1:
            continue

        # identify head and shoulders
        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        headidx = np.argmax(maxim, axis=0)
        # if the headidx is greater than its left maximum and right maximum
        # slope <= 0 almost
        # left local min is on the right of the left shoulder
        # right local min is on the left of the right shoulder
        if (maxim[headidx]-maxim[headidx-1]>1.5e-3 and 
            maxim[headidx]-maxim[headidx+1]>1.5e-3 and 
            abs(slmin)<=1e-4 and 
            xxmin[0]>xxmax[headidx-1] and 
            xxmin[1]<xxmax[headidx+1]):
            # and (maxim[headidx]-maxim[headidx+1])>(maxim[headidx+1]-minim[headidx+1]) and (maxim[headidx]-maxim[headidx-1])>(maxim[headidx-1]-minim[headidx-1]) :
            print("minbcount,minacount,maxbcount,maxacount,slmin,candleid:\n")
            print(minbcount,minacount,maxbcount,maxacount,slmin,candleid)
            #print(maxim)
            #print(xxmax)
            #print(minim)
            #print(xxmin)
            break

        if candleid % 1000 == 0:
            print(candleid)








if __name__ == "__main__":
    df = read_data()
    df = add_pivot(df)
    df = add_point_pos(df)
    plot_signal(df, 8000, 10000)
    back_candles = 14
    detect_head_and_shoulders(df, back_candles)
