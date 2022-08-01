# https://medium.com/@dannygrovesn7/creating-stock-market-trend-lines-in-35-lines-of-python-865906d5ecef

from get_trend_line import find_grad_intercept
import yfinance as yf
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd

pio.renderers.default = 'browser'


# df = yf.download('TSLA').reset_index()
df = pd.read_csv('data/Binance_BTCUSDT_d.csv', usecols=[
    'date', 'symbol', 'open', 'high', 'low', 'close'], skiprows=1)[::-1]

trend_line_df = df

# Perform the date filtering to get the plotting df, and the df to get the
# trend line of
# df = (
#     df[(df['date'] > '2020-05-01') & (df['date'] < '2020-10-01')]
#     .reset_index(drop=True)
# )
# trend_line_df = df[(df['date'] > '2020-07-13') & (df['date'] < '2020-08-11')]

trend_line_df = df[(df['date'] > '2021-07-13') & (df['date'] < '2021-08-11')]

# Using the trend-line algorithm, deduce the gradient and intercept terms of
# the straight lines
m_res, c_res = find_grad_intercept(
    'resistance',
    trend_line_df.index.values,
    trend_line_df.high.values,
)
m_supp, c_supp = find_grad_intercept(
    'support',
    trend_line_df.index.values,
    trend_line_df.low.values,
)

# Plot the figure with plotly
layout = go.Layout(
    title='Some asset',
    xaxis={'title': 'Date'},
    yaxis={'title': 'Price'},
)

fig = go.Figure(
    layout=layout,
    data=[
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick chart'
        ),
        go.Scatter(
            x=trend_line_df['date'],
            y=m_res*trend_line_df.index + c_res,
            name='Resistance line'
        ),
        go.Scatter(
            x=trend_line_df['date'],
            y=m_supp*trend_line_df.index + c_supp,
            name='Support line'
        ),
    ]
)


fig.update_xaxes(
    rangeslider_visible=False,
    rangebreaks=[{'bounds': ['sat', 'mon']}]
)
fig.show()
