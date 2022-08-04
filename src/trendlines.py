# https://medium.com/@dannygrovesn7/creating-stock-market-trend-lines-in-35-lines-of-python-865906d5ecef

from datetime import date, timedelta


import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from get_trend_line import find_grad_intercept
from constants.constants import BINANCE_BTCUSDT_DAY, FORMAT_STR


pio.renderers.default = 'browser'


def get_layout() -> go.Layout:
    """Returns a layout for plotly"""
    return go.Layout(
        title='ABC/USD',
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price'},
    )


def get_config() -> dict:
    return {'modeBarButtonsToAdd': ['drawline',
                                    'drawopenpath',
                                    'drawclosedpath',
                                    'eraseshape',
                                    ],
            'scrollZoom': True,
            'doubleClickDelay': 1000,  # double click the graph to reset position
            'displayModeBar': True,
            }


def normalize_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLC data from 0 to 100
        normalization formula: (data - min) / (max - min)
    """

    _max = np.max([np.max(df['open']), np.max(df['high']),
                  np.max(df['low']), np.max(df['close'])])
    _min = np.min([np.min(df['open']), np.min(df['high']),
                  np.min(df['low']), np.min(df['close'])])

    norm_open = (df['open'] - _min) / (_max - _min)
    norm_high = (df['high'] - _min) / (_max - _min)
    norm_low = (df['low'] - _min) / (_max - _min)
    norm_close = (df['close'] - _min) / (_max - _min)

    mult = 100
    d = df['date']
    o = round(norm_open*mult, 2)
    h = round(norm_high*mult, 2)
    l = round(norm_low*mult, 2)
    c = round(norm_close*mult, 2)
    return pd.DataFrame({'date': d, 'open': o, 'high': h, 'low': l, 'close': c})


def drop_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Returns a dataframe that has dropped the data outside of the start and end dates"""
    df = (df[(df['date'] > start_date) & (df['date'] < end_date)].reset_index(drop=True)
          )
    return df


def main() -> None:
    df = pd.read_csv(BINANCE_BTCUSDT_DAY, usecols=[
        'date', 'symbol', 'open', 'high', 'low', 'close'], skiprows=1)[::-1]
    df = normalize_ohlc_data(df)

    # support and resistance lines will be 'delta' bars long
    delta = timedelta(days=14)

    # 2021-12-3 -> 2021-12-16
    draw_start_date_i = date(2021, 12, 3)
    draw_end_date_i = date(2021, 12, 16)

    draw_start_date_i = date(2021, 1, 1)
    draw_end_date_i = date(2022, 1, 1)

    # create the dates to draw the s/r lines
    end_date_j = draw_start_date_i + delta
    # end_date_inner = draw_start_date_i + delta

    m_res = None
    m_supp = None
    c_res = None
    c_supp = None
    trend_line_df = None

    graph_start_date = (draw_start_date_i - delta * 10).strftime(FORMAT_STR)
    graph_end_date = draw_end_date_i.strftime(FORMAT_STR)

    # drop data we are not looking at
    df = drop_data(df, graph_start_date, graph_end_date)

    data = [
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick chart',
        ),

    ]

    # iterate over all the dates we want to draw support and resistance lines
    while draw_start_date_i <= draw_end_date_i:
        # while end_date_j <= end_date_inner:
        trend_line_df = df[(df['date'] > draw_start_date_i.strftime(FORMAT_STR)) &
                           (df['date'] < end_date_j.strftime(FORMAT_STR))]

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

        # add the dates to draw the s/r lines
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

        # change the start date to the next delta after each iteration
        draw_start_date_i += delta

        # end_date_inner = draw_start_date_i + delta

    fig = go.Figure(layout=get_layout(), data=data, )
    fig.update_xaxes(rangeslider_visible=False)

    # fig.add_annotation(
    #     x=0.5,
    #     y=0.5,
    #     text=text,
    #     xref="paper",
    #     yref="paper",
    #     showarrow=False,
    #     font_size=20
    # )

    # define dragmode and add modebar buttons
    fig.update_layout(dragmode='zoom', newshape_line_color='salmon')
    # fig.update_traces(line_color='red', selector=dict(type='scatter'))
    fig.show(config=get_config())
    fig.write_html('html/ABC-USD.html')


# TODO:
    # Create a google document like page with plotly integrated
    # only show 1 graph per page
    # do not allow the user to go forward or backward
    # Integrate drawing tools and have all the data that the users plots on the graph be saved when they continue onto the next graph.
    # Do not reveal any answers until the test is complete.
    # Create a time limit for the entire test (maybe even for each graph?)
    # Create additional notes section at the bottom to allow the user to say anything they want. This will have no effect on score

    # Checkbox: real or fake
    # Checkbox: Skip -> offer this because the user cannot see a pattern but dock them points if they do so.
    # confidence slider: 10% increments
    # price prediction input for 1 bar, 5, bars, 10 bars, 20 bars in the future
    # The farther out, the user can predict, the more points are awarded.
    # Give the user a choice for an exact number or for a lower limit and upper limit.
    # The more precise the user is, the more point are awarded. The less precise, the less points are awarded.


if __name__ == '__main__':
    main()
