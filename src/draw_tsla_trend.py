# https://gist.github.com/GrovesD2/6e95320af2fd9455f8743602280c2442#file-draw_tsla_trend-py

from get_trend_line import find_grad_intercept
import yfinance as yf
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'browser'


df = yf.download('TSLA').reset_index()

# Perform the date filtering to get the plotting df, and the df to get the
# trend line of
df = (
    df[(df['Date'] > '2020-05-01') & (df['Date'] < '2020-10-01')]
    .reset_index(drop=True)
)
trend_line_df = df[(df['Date'] > '2020-07-13') & (df['Date'] < '2020-08-11')]

# Using the trend-line algorithm, deduce the gradient and intercept terms of
# the straight lines
m_res, c_res = find_grad_intercept(
    'resistance',
    trend_line_df.index.values,
    trend_line_df.High.values,
)
m_supp, c_supp = find_grad_intercept(
    'support',
    trend_line_df.index.values,
    trend_line_df.Low.values,
)

# Plot the figure with plotly
layout = go.Layout(
    title='TSLA Stock Price',
    xaxis={'title': 'Date'},
    yaxis={'title': 'Price'},
)

fig = go.Figure(
    layout=layout,
    data=[
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick chart'
        ),
        go.Scatter(
            x=trend_line_df['Date'],
            y=m_res*trend_line_df.index + c_res,
            name='Resistance line'
        ),
        go.Scatter(
            x=trend_line_df['Date'],
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
