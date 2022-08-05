import random
import datetime
from pprint import pprint
from typing import Tuple

from dash import Dash, dcc, html
import pandas as pd
import plotly.graph_objects as go  # or plotly.express as px
import numpy as np

from constants.constants import *
from RandomOHLC import RandomOHLC
from RealOHLC import RealOHLC


data_date_ranges = {
    # spot
    BINANCE_BTCUSDT_DAY: {'start_date': '2019-09-08', 'end_date': '2022-06-16'},
    BINANCE_AAVEUSDT_DAY: {'start_date': '2020-10-16', 'end_date': '2022-06-16'},
    BINANCE_ADAUSDT_DAY: {'start_date': '2018-04-17', 'end_date': '2022-07-30'},
    BINANCE_CELRUSDT_DAY: {'start_date': '2019-03-25', 'end_date': '2022-07-30'},
    BINANCE_DASHUSDT_DAY: {'start_date': '2019-03-28', 'end_date': '2022-07-30'},
    BINANCE_DOGEUSDT_DAY: {'start_date': '2020-07-10', 'end_date': '2022-07-30'},
    BINANCE_DOTUSDT_DAY: {'start_date': '2020-08-18', 'end_date': '2022-07-30'},
    BINANCE_ETCUSDT_DAY: {'start_date': '2018-06-12', 'end_date': '2022-07-30'},
    BINANCE_ETHUSDT_DAY: {'start_date': '2017-08-17', 'end_date': '2022-07-30'},

    # spot
    BINANCE_ETHUSDT_FUTURES_DAY: {'start_date': '2019-11-27', 'end_date': '2022-03-15'},
    BINANCE_LTCUSDT_FUTURES_DAY: {'start_date': '2020-01-09', 'end_date': '2022-03-15'},
    BINANCE_ADAUSDT_FUTURES_DAY: {'start_date': '2020-01-31', 'end_date': '2022-07-30'},
    BINANCE_BTCUSDT_FUTURES_DAY: {'start_date': '2019-09-08', 'end_date': '2022-03-15'},
    BINANCE_XMRUSDT_FUTURES_DAY: {
        'start_date': '2020-02-03', 'end_date': '2022-07-30'}
}


app = Dash()


def app_update_layout(fig: go.Figure) -> html.Div:
    return html.Div([
        dcc.Graph(figure=fig,
                  config={
                      'doubleClickDelay': 1000,
                      'scrollZoom': True,
                      'displayModeBar': True,
                      'showTips': True,
                      'displaylogo': True,
                    #   'fillFrame': True,
                      'autosizable': True,
                      'modeBarButtonsToAdd': ['drawline',
                                              'drawopenpath',
                                              'drawclosedpath',
                                              'eraseshape',
                                              ],
                    }
                  )])


def normalize_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
    """

    _max = np.max([np.max(df.open), np.max(df.high),
                  np.max(df.low), np.max(df.close)])
    _min = np.min([np.min(df.open), np.min(df.high),
                  np.min(df.low), np.min(df.close)])

    norm_open = (df.open - _min) / (_max - _min)
    norm_high = (df.high - _min) / (_max - _min)
    norm_low = (df.low - _min) / (_max - _min)
    norm_close = (df.close - _min) / (_max - _min)

    random_multiplier = random.randint(9, 999)

    df['open'] = round(norm_open*random_multiplier, 4)
    df['high'] = round(norm_high*random_multiplier, 4)
    df['low'] = round(norm_low*random_multiplier, 4)
    df['close'] = round(norm_close*random_multiplier, 4)
    return df


def get_date_limits(days: int, data_choice: int) -> Tuple[str, str]:
    d_range = data_date_ranges.get(data_choice)
    start_date__limit_l = [
        int(i) for i in d_range['start_date'].split('-')]

    # adjust the start_date 91 days after the original start date
    start_dt_limit = datetime.datetime(
        year=start_date__limit_l[0], month=start_date__limit_l[1], day=start_date__limit_l[2])
    adjusted_start_dt = start_dt_limit + datetime.timedelta(days=days)
    adj_start_date_limit = adjusted_start_dt.strftime("%Y-%m-%d")
    end_date_limit = d_range['end_date']
    return adj_start_date_limit, end_date_limit


def create_dates(num_days_range: int, start_date_limit: str, end_date_limit: str) -> tuple[str, str]:
    """Randomly pick a start and end date within the given starting and ending bounds"""

    start_date_limit_l = [int(i) for i in start_date_limit.split('-')]
    end_date_limit_l = [int(i) for i in end_date_limit.split('-')]

    start_limit_dt = datetime.datetime(year=start_date_limit_l[0],
                                       month=start_date_limit_l[1],
                                       day=start_date_limit_l[2])

    end_limit_dt = datetime.datetime(year=end_date_limit_l[0],
                                     month=end_date_limit_l[1],
                                     day=end_date_limit_l[2])

    # get the number of days from the start date to the end date
    date_range_limit = end_limit_dt - start_limit_dt

    # create a list of all the dates within the given date bounds
    dt_list = [start_limit_dt +
               datetime.timedelta(days=x) for x in range(date_range_limit.days)]

    # pick a random day to start minus the given range
    start_i = random.randint(0, len(dt_list) - num_days_range)
    end_i = start_i + num_days_range

    start_random_dt = dt_list[start_i]
    end_random_dt = dt_list[end_i]
    return start_random_dt.strftime("%Y-%m-%d"), end_random_dt.strftime("%Y-%m-%d")


def real_case(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Create a dataframe for real data"""
    df = df.drop(df[df['date'] < start_date].index)
    df = df.drop(df[df['date'] > end_date].index)

    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df.set_index('date', inplace=True)
    return df


def create_half_df(df: pd.DataFrame) -> pd.DataFrame:
    df['open'] = df['open'].iloc[:len(df.open)//2]
    df['high'] = df['high'].iloc[:len(df.high)//2]
    df['low'] = df['low'].iloc[:len(df.low)//2]
    df['close'] = df['close'].iloc[:len(df.close)//2]
    return df


def create_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Candlestick(
        x=df.index,
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
    ))

    fig.update_layout(
        template='plotly_dark',
        # title=answer,
        xaxis_title="Time",
        yaxis_title="Value",
        dragmode='zoom',
        newshape_line_color='white',

        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ),

        # hides the xaxis range slider
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        ),

    )

    # fig.update_yaxes(showticklabels=True)
    # fig.update_xaxes(showticklabels=True)
    return fig


def main() -> None:
    start_price = 100_000
    asset_name = 'Unknown'
    total_graphs = 5
    num_days_range = 120
    answers = {}
    global app

    for i in range(total_graphs):
        df = None
        # pick dates randomly
        data_choice = random.choice(list(data_date_ranges.keys()))
        adj_start_date_limit, end_date_limit = get_date_limits(
            91, data_choice)

        start_date, end_date = create_dates(
            num_days_range, adj_start_date_limit, end_date_limit)

        if random.randint(0, 1):
            # if False:
            real_ohlc = RealOHLC(data_choice)
            df = real_ohlc.create_df()
            df = real_ohlc.real_case(df, start_date, end_date)
            answers[i] = f"Real: {start_date} to {end_date} {data_choice}"
        else:
            random_ohlc = RandomOHLC(num_days_range, start_price,
                                     start_date, asset_name)
            df = random_ohlc.create_random_df()
            df = random_ohlc.create_realistic_candles(df)
            answers[i] = f"Fake"

        df = normalize_ohlc_data(df)
        df.reset_index(inplace=True)
        df.drop(columns=['date'], inplace=True)

        # create a new df that contains only half the dates and prices
        half_df = df.iloc[:len(df)//2]

        fig = create_figure(half_df)
        fig.write_html(f"html/HABC-USD_{i}.html")
        app.layout = app_update_layout(fig)

        # delete when done testing!
        # fig.show()

        # This is the full graph that only the admin should be able to see!
        ####################################################################
        # fig = create_figure(df.index, norm_open, norm_high,
        #                     norm_low, norm_close, f'FABC/USD {i}')
        # fig.write_html(f"html/FABC/USD {i}.html")
        # fig.show()
        ####################################################################

    app.run_server(debug=True, use_reloader=True)
    pprint(answers)


if __name__ == '__main__':
    main()
