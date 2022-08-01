# https://medium.com/@dannygrovesn7/creating-stock-market-trend-lines-in-35-lines-of-python-865906d5ecef

from datetime import date, timedelta

import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd

from get_trend_line import find_grad_intercept

pio.renderers.default = 'browser'


df = pd.read_csv('data/Binance_BTCUSDT_d.csv', usecols=[
    'date', 'symbol', 'open', 'high', 'low', 'close'], skiprows=1)[::-1]


# Perform the date filtering to get the plotting df, and the df to get the
# df = (
#     df[(df['date'] > '2020-11-01') & (df['date'] < '2022-01-01')]
#     .reset_index(drop=True)
# )

# Plot the figure with plotly
layout = go.Layout(
    title='Some asset',
    xaxis={'title': 'Date'},
    yaxis={'title': 'Price'},
)

data = [
    go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick chart'
    ),
]

delta = timedelta(days=14)
start_date_i = date(2022, 1, 1)
end_date_i = date(2022, 7, 21)
end_date_j = start_date_i + delta
end_date_inner = start_date_i + delta

m_res = None
m_supp = None
c_res = None
c_supp = None
trend_line_df = None


while start_date_i <= end_date_i:
    while end_date_j <= end_date_inner:
        trend_line_df = df[(df['date'] > start_date_i.strftime("%Y-%m-%d")) &
                           (df['date'] < end_date_j.strftime("%Y-%m-%d"))]

        # Using the trend-line algorithm, deduce the
        # gradient and intercept terms of the straight lines
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

        end_date_j += delta

    data.append(
        go.Scatter(
            x=trend_line_df['date'],
            y=m_res*trend_line_df.index + c_res,
            name='Resistance line '
        ),
    )
    data.append(
        go.Scatter(
            x=trend_line_df['date'],
            y=m_supp*trend_line_df.index + c_supp,
            name='Support line'
        ),
    )

    start_date_i += delta
    end_date_inner = start_date_i + delta

fig = go.Figure(
    layout=layout,
    data=data
)

fig.update_xaxes(rangeslider_visible=False)
fig.show()
