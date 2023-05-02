"""
Channel Break-Out Detection
    - Pivot points detection
    - Channel dectection
    - Break-Out detection
Detect pivots, then fit two lines (upper = resistance, lower = support) 
with those pivots to form a channel
Detect breakout points relative to the channel to indicate future trends
The intuition is based on Momentum
"""



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats



def read_data_eurousd():
    df = pd.read_csv("EURUSD_Candlestick_1_D_BID_04.05.2003-21.01.2023.csv")
    return df


def isPivot(candle, window):
    """
    function that detects if a candle is a pivot/fractal point
    args: candle index, window before and after candle to test if pivot
    returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
    """
    # if out of range, return 0
    if candle - window < 0 or candle + window >= len(df):
        return 0

    pivotHigh = 1
    pivotLow = 2
    for i in range(candle-window, candle+window+1):
        # if there exists a candle in the window
        # whose low price is lower than candle's low price,
        # the candle is not a low pivot
        if df.iloc[candle].Low > df.iloc[i].Low:
            pivotLow = 0
        # if there exists a candle in the window
        # whose high price is greater than candle's high price,
        # the candle is not a high pivot
        if df.iloc[candle].High < df.iloc[i].High:
            pivotHigh = 0
    # a candle can be both a high pivot and a low pivot
    if (pivotHigh and pivotLow):
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0


def detect_pivot(df, window):
    df['isPivot'] = df.apply(lambda x: isPivot(x.name,window), axis=1)
    return df


def get_point_pos(df):
    if df['isPivot']==2:
        return df['Low']-1e-3
    elif df['isPivot']==1:
        return df['High']+1e-3
    else:
        return np.nan


def add_point_pos(df):
    df['pointpos'] = df.apply(lambda row: get_point_pos(row), axis=1)
    return df


def create_and_plot_pivot(df):
    dfpl = df[0:100]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()
    return dfpl


def collect_channel(df, candle, back_candles, window):
    """
    collect price channel
    avoid the <look-ahead bias>
    """
    # slicing the df from candle - back_candles - window
    #                to   candle - window
    # since we cannot know the future, we cannot look ahead,
    # in order for our method to make sense,
    # we need to look at candle - window instead of candle
    localdf = df[candle - back_candles - window : candle - window]
    localdf['isPivot'] = localdf.apply(lambda x: isPivot(x.name,window), axis=1)
    highs = localdf[localdf['isPivot']==1].High.values
    idx_highs = localdf[localdf['isPivot']==1].High.index
    lows = localdf[localdf['isPivot']==2].Low.values
    idx_lows = localdf[localdf['isPivot']==2].Low.index

    # if we have more than two low points and more than two high points
    # we create a channel
    if len(lows)>=2 and len(highs)>=2:
        # slope, intercept, r_value, ...
        sl_lows, interc_lows, r_value_l, _, _ = \
            stats.linregress(idx_lows,lows)
        # slope, intercept, r_value, ...
        sl_highs, interc_highs, r_value_h, _, _ = \
            stats.linregress(idx_highs,highs)
        # R-Squared (the coefficient of determination) is a statistical measure in a regression model that determines the proportion of variance in the dependent variable that can be explained by the independent variable.
        return(sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
    else:
        return(0,0,0,0,0,0)


def plot_channel(df, dfpl, candle, back_candles, window):
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(df, candle, back_candles, window)
    print(r_sq_l, r_sq_h)
    x = np.array(range(candle-back_candles-window, candle+1))
    fig.add_trace(go.Scatter(x=x, y=sl_lows*x +
                             interc_lows, mode='lines',
                             name='support'))
    fig.add_trace(go.Scatter(x=x, y=sl_highs*x +
                             interc_highs, mode='lines',
                             name='resistance'))
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()


def isBreakOut(df, candle, back_candles, window):
    """
    determine whether a candle indicates a breakout
    """
    if (candle-back_candles-window)<0:
        return 0

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(df, candle, back_candles, window)

    # previous candle
    prev_idx = candle-1
    prev_high = df.iloc[candle-1].High
    prev_low = df.iloc[candle-1].Low
    prev_close = df.iloc[candle-1].Close

    # current candle
    curr_idx = candle
    curr_high = df.iloc[candle].High
    curr_low = df.iloc[candle].Low
    curr_close = df.iloc[candle].Close
    curr_open = df.iloc[candle].Open

    # bearish breakout
    # if the previous high price is above the lower channel line,
    # i.e., in the channel
    # and the previous close price is below the lower channel line,
    # and the current open and current close prices are
    # below than the lower channel line
    # we forecast a future downtrend
    if ( prev_high > (sl_lows*prev_idx + interc_lows) and
        prev_close < (sl_lows*prev_idx + interc_lows) and
        curr_open < (sl_lows*curr_idx + interc_lows) and
        curr_close < (sl_lows*prev_idx + interc_lows)): #and r_sq_l > 0.9
        return 1
    # bullish breakout
    # if the previous low price is below the upper channel line,
    # i.e., in the channel
    # and the previous close price is above the upper channel line,
    # and the current open and current close prices are
    # above than the upper channel line
    # we forecast a future uptrend
    elif ( prev_low < (sl_highs*prev_idx + interc_highs) and
        prev_close > (sl_highs*prev_idx + interc_highs) and
        curr_open > (sl_highs*curr_idx + interc_highs) and
        curr_close > (sl_highs*prev_idx + interc_highs)): #and r_sq_h > 0.9
        return 2
    # not a breakout point
    else:
        return 0


def get_breakout_point_pos(df):
    """
    The intuition here conflicts with the isBreakOut()
    """
    # bullish breakout candle
    # future uptrend
    # low is the best buyin position
    if df['isBreakOut']==2:
        return df['Low']-3e-3
    # bearish breakout candle
    # future downtrend
    # high is the best sellout position
    elif df['isBreakOut']==1:
        return df['High']+3e-3
    else:
        return np.nan


def add_breakout_point_pos(df, candle, back_candles, window):
    dfpl = df[candle-back_candles-window-5:candle+20]
    dfpl["isBreakOut"] = [isBreakOut(df, candle, back_candles, window) for candle in dfpl.index]
    dfpl['breakpointpos'] = dfpl.apply(lambda row: get_breakout_point_pos(row), axis=1)
    return dfpl


def plot_breakout(df, dfpl, candle, back_candles, window):
    """
    The graph is a little bit confusing
    Focus on the candle position, ignore any breakout hexagram before it
    """
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")

    fig.add_scatter(x=dfpl.index, y=dfpl['breakpointpos'], mode="markers",
                    marker=dict(size=8, color="Black"), marker_symbol="hexagram",
                    name="breakout")

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(df, candle, back_candles, window)
    print(r_sq_l, r_sq_h)
    x = np.array(range(candle-back_candles-window, candle+1))
    fig.add_trace(go.Scatter(x=x, y=sl_lows*x + interc_lows, mode='lines', name='support'))
    fig.add_trace(go.Scatter(x=x, y=sl_highs*x + interc_highs, mode='lines', name='resistance'))
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()






if __name__ == "__main__":
    df = read_data_eurousd()

    candle = 75
    back_candles = 40
    window = 3

    df = detect_pivot(df, window)
    df = add_point_pos(df)

    dfpl = create_and_plot_pivot(df)
    # plot_channel(df, dfpl, candle, back_candles, window)
    dfpl = add_breakout_point_pos(df, candle, back_candles, window)
    plot_breakout(df, dfpl, candle, back_candles, window)
