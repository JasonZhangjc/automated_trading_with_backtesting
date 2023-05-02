"""
Detect triangle pattern
    - Descending triangle
    - Ascending triangle
    - Converging triangle
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from matplotlib import pyplot
from scipy.stats import linregress





def read_data():
    df = pd.read_csv("EURUSD_Candlestick_4_Hour_ASK_05.05.2003-16.10.2021.csv")
    df.columns=['time', 'open', 'high', 'low', 'close', 'volume']
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df.isna().sum()
    return df



def add_pivot(df):
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
    # relatively short pivot
    df['pivot'] = df.apply(lambda x: pivotid(df, x.name,3,3), axis=1)
    return df



def add_point_pos(df):
    def get_point_pos(x):
        if x['pivot']==1:
            return x['low']-1e-3
        elif x['pivot']==2:
            return x['high']+1e-3
        else:
            return np.nan
    df['pointpos'] = df.apply(lambda row: get_point_pos(row), axis=1)
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
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()



def linear_regression(df, candleid, back_candles):
    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])

    for i in range(candleid-back_candles, candleid+1):
        if df.iloc[i].pivot == 1:
            minim = np.append(minim, df.iloc[i].low)
            xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
        if df.iloc[i].pivot == 2:
            maxim = np.append(maxim, df.iloc[i].high)
            xxmax = np.append(xxmax, i) # df.iloc[i].name

    #slmin, intercmin = np.polyfit(xxmin, minim,1) #numpy
    #slmax, intercmax = np.polyfit(xxmax, maxim,1)

    slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
    slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)

    print(rmin, rmax)

    dfpl = df[candleid-back_candles-10:candleid+back_candles+10]

    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['open'],
                    high=dfpl['high'],
                    low=dfpl['low'],
                    close=dfpl['close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=4, color="MediumPurple"),
                    name="pivot")

    #-------------------------------------------------------------------------
    # Fitting intercepts to meet highest or lowest candle point in time slice
    #adjintercmin = df.low.loc[candleid-backcandles:candleid].min() - slmin*df.low.iloc[candleid-backcandles:candleid].idxmin()
    #adjintercmax = df.high.loc[candleid-backcandles:candleid].max() - slmax*df.high.iloc[candleid-backcandles:candleid].idxmax()

    xxmin = np.append(xxmin, xxmin[-1]+15)
    xxmax = np.append(xxmax, xxmax[-1]+15)
    #fig.add_trace(go.Scatter(x=xxmin, y=slmin*xxmin + adjintercmin, mode='lines', name='min slope'))
    #fig.add_trace(go.Scatter(x=xxmax, y=slmax*xxmax + adjintercmax, mode='lines', name='max slope'))

    fig.add_trace(go.Scatter(x=xxmin, y=slmin*xxmin + intercmin, mode='lines', name='min slope'))
    fig.add_trace(go.Scatter(x=xxmax, y=slmax*xxmax + intercmax, mode='lines', name='max slope'))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()



def triangle_pattern(df, start, back_candles):
    for candleid in range(start, len(df)):
        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])
        for i in range(candleid-back_candles, candleid+1):
            if df.iloc[i].pivot == 1:
                minim = np.append(minim, df.iloc[i].low)
                xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
            if df.iloc[i].pivot == 2:
                maxim = np.append(maxim, df.iloc[i].high)
                xxmax = np.append(xxmax, i) # df.iloc[i].name

        if (xxmax.size <3 and xxmin.size <3) or xxmax.size==0 or xxmin.size==0:
            continue

        #slmin, intercmin = np.polyfit(xxmin, minim,1) #numpy
        #slmax, intercmax = np.polyfit(xxmax, maxim,1)

        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)

        # descending triangle
        #if abs(rmax)>=0.7 and abs(rmin)>=0.7 and abs(slmin)<=0.00001 and slmax<-0.0001:

        # ascending triangle
        #if abs(rmax)>=0.7 and abs(rmin)>=0.7 and slmin>=0.0001 and abs(slmax)<=0.00001:

        # detect the triangle
        # abs(r) is close to 1 if it is very likely a linear line
        # whether linear regression is suitable for the data points
        if (abs(rmax)>=0.9 and abs(rmin)>=0.9 and 
            slmin>=0.0001 and slmax<=-0.0001):
            print(rmin, rmax, candleid)
            break

        if candleid % 1000 == 0:
            print(candleid)









if __name__ == "__main__":
    df = read_data()
    df = add_pivot(df)
    df = add_point_pos(df)
    plot_signal(df, 28000, 28500)
    back_candles = 20
    candleid = 4696
    linear_regression(df, candleid, back_candles)
    triangle_pattern(df, 11000, back_candles)