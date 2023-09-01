# Flag pattern 
# Need to use linear regression


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import linregress



def read_data():
    df = pd.read_csv("../data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv")
    df.columns=['time', 'open', 'high', 'low', 'close', 'volume']
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df.isna().sum()
    # df.head(10)

    return df



def pivotid(df1, l, n1, n2): #n1 n2 before and after candle l
    # l: tested candle
    # n1: number of candles before l
    # n2: number of candles after l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0

    pividlow=1
    pividhigh=1
    for i in range(l-n1, l+n2+1):
        if(df1.low[l]>df1.low[i]):
            pividlow=0   # not low anymore
        if(df1.high[l]<df1.high[i]):
            pividhigh=0  # not high anymore
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0



def apply_pivot(df):
    df['pivot'] = df.apply(lambda x: pivotid(df, x.name,3,3), axis=1)

    return df



def pointpos(x):
    if x['pivot']==1:
        return x['low']-1e-3
    elif x['pivot']==2:
        return x['high']+1e-3
    else:
        return np.nan
    


def apply_pointpos(df):
    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

    return df



def visualize_pointpos(df):
    dfpl = df[500:800]
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



def detect_flag(candle, backcandles, window, plot_flag=False):
    """
    Attention! window should always be greater than the pivot window! to avoid look ahead bias
    window is used to avoid look ahead bias
    window > 0
    """
    localdf = df[candle-backcandles-window: candle-window]
    highs = localdf[localdf['pivot'] == 2].high.tail(3).values
    idxhighs = localdf[localdf['pivot'] == 2].high.tail(3).index
    lows = localdf[localdf['pivot'] == 1].low.tail(3).values
    idxlows = localdf[localdf['pivot'] == 1].low.tail(3).index

    if len(highs) == 3 and len(lows) == 3:
        order_condition = (
            (idxlows[0] < idxhighs[0]
            < idxlows[1] < idxhighs[1]
            < idxlows[2] < idxhighs[2])
            or
            (idxhighs[0] < idxlows[0]
             < idxhighs[1] < idxlows[1]
             < idxhighs[2] < idxlows[2]) )

        # linear regression
        slmin, intercmin, rmin, _, _ = linregress(idxlows, lows)
        slmax, intercmax, rmax, _, _ = linregress(idxhighs, highs)

        # rmax*rmax or rmin*rmin measures how linear it is
        # the larger, the linear
        if (order_condition
            and (rmax*rmax)>=0.9
            and (rmin*rmin)>=0.9
            and slmin>=0.0001
            and slmax<=-0.0001):
            #and ((abs(slmin)-abs(slmax))/abs(slmax)) < 0.05):

            if plot_flag:
                fig = go.Figure(data=[go.Candlestick(x=localdf.index,
                open=localdf['open'],
                high=localdf['high'],
                low=localdf['low'],
                close=localdf['close'])])

                fig.add_scatter(x=localdf.index, y=localdf['pointpos'], mode="markers",
                marker=dict(size=10, color="MediumPurple"),
                name="pivot")
                fig.add_trace(go.Scatter(x=idxlows, y=slmin*idxlows + intercmin, mode='lines', name='min slope'))
                fig.add_trace(go.Scatter(x=idxhighs, y=slmax*idxhighs + intercmax, mode='lines', name='max slope'))
                fig.update_layout(
                xaxis_rangeslider_visible=False,
                plot_bgcolor='white', # change the background to white
                xaxis=dict(showgrid=True, gridcolor='white'), # change the x-axis grid to white
                yaxis=dict(showgrid=True, gridcolor='white') # change the y-axis grid to white
                )
                fig.show()

            return 1

    return 0



def apply_flag(df):
    df['flag'] = df.index.map(lambda x: detect_flag(x, 35, 3))

    return df




if __name__ == "__main__":
    df = read_data()
    df = apply_pivot(df)
    df = apply_pointpos(df)
    visualize_pointpos(df)
    df = apply_flag(df)
    detect_flag(2189, 30, 3, plot_flag=True)
