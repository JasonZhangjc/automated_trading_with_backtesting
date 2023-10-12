import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt





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
        return 3   # undetermined
    elif pividlow:
        return 1   # low = 1
    elif pividhigh:
        return 2   # high = 2
    else:
        return 0   # undertermined



def pointpos(x):
    if x['pivot']==1:
        return x['low']-1e-3
    elif x['pivot']==2:
        return x['high']+1e-3
    else:
        return np.nan



def plot_signal(df):
    dfpl = df[-300:-1]
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['open'],
                    high=dfpl['high'],
                    low=dfpl['low'],
                    close=dfpl['close'],
                    increasing_line_color= 'green', 
                    decreasing_line_color= 'red')])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')

    fig.show()



def plot_value(df):
    dfkeys = df[:]

    # Filter the dataframe based on the pivot column
    high_values = dfkeys[dfkeys['pivot'] == 2]['high']
    low_values = dfkeys[dfkeys['pivot'] == 1]['low']

    # Define the bin width
    bin_width = 0.003  # Change this value as needed

    # Calculate the number of bins
    bins = int((high_values.max() - low_values.min()) / bin_width)

    # Create the histograms
    plt.figure(figsize=(10, 5))
    plt.hist(high_values, bins=bins, alpha=0.5, label='High Values', color='red')
    plt.hist(low_values, bins=bins, alpha=0.5, label='Low Values', color='blue')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of High and Low Values')
    plt.legend()

    plt.show()





if __name__ == "__main__":
    df = read_data()
    df['pivot'] = df.apply(lambda x: pivotid(df, x.name,10,10), axis=1)
    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    plot_signal(df)
    plot_value(df)