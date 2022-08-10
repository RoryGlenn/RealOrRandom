import random
import datetime
from pprint import pprint
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

from constants.constants import *
from random_ohlc import RandomOHLC
from real_ohlc import RealOHLC


def get_data_date_ranges() -> dict:
    """Returns a dictionary will the earliest start date and latest end date we can use for each real data file"""
    return {
        # spot
        BINANCE_BTCUSDT_DAY: {"start_date": "2019-09-08", "end_date": "2022-06-16"},
        BINANCE_AAVEUSDT_DAY: {"start_date": "2020-10-16", "end_date": "2022-06-16"},
        BINANCE_ADAUSDT_DAY: {"start_date": "2018-04-17", "end_date": "2022-07-30"},
        BINANCE_CELRUSDT_DAY: {"start_date": "2019-03-25", "end_date": "2022-07-30"},
        BINANCE_DASHUSDT_DAY: {"start_date": "2019-03-28", "end_date": "2022-07-30"},
        BINANCE_DOGEUSDT_DAY: {"start_date": "2020-07-10", "end_date": "2022-07-30"},
        BINANCE_DOTUSDT_DAY: {"start_date": "2020-08-18", "end_date": "2022-07-30"},
        BINANCE_ETCUSDT_DAY: {"start_date": "2018-06-12", "end_date": "2022-07-30"},
        BINANCE_ETHUSDT_DAY: {"start_date": "2017-08-17", "end_date": "2022-07-30"},
        # spot
        BINANCE_ETHUSDT_FUTURES_DAY: {
            "start_date": "2019-11-27",
            "end_date": "2022-03-15",
        },
        BINANCE_LTCUSDT_FUTURES_DAY: {
            "start_date": "2020-01-09",
            "end_date": "2022-03-15",
        },
        BINANCE_ADAUSDT_FUTURES_DAY: {
            "start_date": "2020-01-31",
            "end_date": "2022-07-30",
        },
        BINANCE_BTCUSDT_FUTURES_DAY: {
            "start_date": "2019-09-08",
            "end_date": "2022-03-15",
        },
        BINANCE_XMRUSDT_FUTURES_DAY: {
            "start_date": "2020-02-03",
            "end_date": "2022-07-30",
        },
    }


def app_update_layout(fig: go.Figure) -> html.Div:
    """Updates the layout for the graph figure"""
    return html.Div(
        [
            dcc.Graph(
                figure=fig,
                config={
                    "doubleClickDelay": 1000,
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "showTips": True,
                    "displaylogo": True,
                    "fillFrame": False,
                    "autosizable": True,
                    "modeBarButtonsToAdd": [
                        "drawline",
                        "drawopenpath",
                        "drawclosedpath",
                        "eraseshape",
                    ],
                },
            )
        ]
    )


def get_config() -> dict:
    return (
        {
            "doubleClickDelay": 1000,
            "scrollZoom": True,
            "displayModeBar": True,
            "showTips": True,
            "displaylogo": True,
            "fillFrame": False,
            "autosizable": True,
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "eraseshape",
            ],
        },
    )


def normalize_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLC data with random multiplier
    normalization formula: (data - min) / (max - min)
    """

    _max = np.max([np.max(df.open), np.max(df.high), np.max(df.low), np.max(df.close)])
    _min = np.min([np.min(df.open), np.min(df.high), np.min(df.low), np.min(df.close)])

    norm_open = (df.open - _min) / (_max - _min)
    norm_high = (df.high - _min) / (_max - _min)
    norm_low = (df.low - _min) / (_max - _min)
    norm_close = (df.close - _min) / (_max - _min)

    random_multiplier = random.randint(9, 999)

    df["open"] = round(norm_open * random_multiplier, 4)
    df["high"] = round(norm_high * random_multiplier, 4)
    df["low"] = round(norm_low * random_multiplier, 4)
    df["close"] = round(norm_close * random_multiplier, 4)
    return df


def get_date_limits(days: int, data_choice: int) -> Tuple[str, str]:
    """Returns the absolute start and end date for a specific data file"""
    d_range = get_data_date_ranges().get(data_choice)
    start_date__limit_l = [int(i) for i in d_range["start_date"].split("-")]

    # adjust the start_date 91 days after the original start date
    start_dt_limit = datetime.datetime(
        year=start_date__limit_l[0],
        month=start_date__limit_l[1],
        day=start_date__limit_l[2],
    )
    adjusted_start_dt = start_dt_limit + datetime.timedelta(days=days)
    adj_start_date_limit = adjusted_start_dt.strftime("%Y-%m-%d")
    end_date_limit = d_range["end_date"]
    return adj_start_date_limit, end_date_limit


def create_dates(
    num_days_range: int, start_date_limit: str, end_date_limit: str
) -> tuple[str, str]:
    """Randomly pick a start and end date within the given starting and ending bounds"""

    start_date_limit_l = [int(i) for i in start_date_limit.split("-")]
    end_date_limit_l = [int(i) for i in end_date_limit.split("-")]

    start_limit_dt = datetime.datetime(
        year=start_date_limit_l[0],
        month=start_date_limit_l[1],
        day=start_date_limit_l[2],
    )

    end_limit_dt = datetime.datetime(
        year=end_date_limit_l[0], month=end_date_limit_l[1], day=end_date_limit_l[2]
    )

    # get the number of days from the start date to the end date
    date_range_limit = end_limit_dt - start_limit_dt

    # create a list of all the dates within the given date bounds
    dt_list = [
        start_limit_dt + datetime.timedelta(days=x)
        for x in range(date_range_limit.days)
    ]

    # pick a random day to start minus the given range
    start_i = random.randint(0, len(dt_list) - num_days_range)
    end_i = start_i + num_days_range

    start_random_dt = dt_list[start_i]
    end_random_dt = dt_list[end_i]
    return start_random_dt.strftime("%Y-%m-%d"), end_random_dt.strftime("%Y-%m-%d")


def create_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Candlestick(
            x=df.index,
            open=df.open,
            high=df.high,
            low=df.low,
            close=df.close,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        # title=answer,
        xaxis_title="Time",
        yaxis_title="Value",
        dragmode="zoom",
        newshape_line_color="white",
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        # hides the xaxis range slider
        xaxis=dict(rangeslider=dict(visible=True)),
    )

    # fig.update_yaxes(showticklabels=True)
    # fig.update_xaxes(showticklabels=True)
    return fig


def create_half_dataframes(
    dataframes: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Creates a new dict that contains only the first half the data in the dataframes"""
    # for timeframe, df in dataframes.items():
    #     dataframes[timeframe] = df.iloc[: len(df) // 2]
    # return dataframes
    return {
        timeframe: dataframes[timeframe].iloc[: len(dataframes[timeframe]) // 2]
        for timeframe in dataframes
    }


def main() -> None:
    start_price = 100_000
    asset_name = "Unknown"
    total_graphs = 1
    num_days_range = 120  # 120 will be the standard
    volatility = random.uniform(1, 2)
    answers = {}
    days = 91
    app = Dash()

    for i in range(total_graphs):
        dataframes = None
        half_dataframes = None
        data_choice = random.choice(list(get_data_date_ranges().keys()))

        # if random.randint(0, 1):
        if False:
            adj_start_date_limit, end_date_limit = get_date_limits(days, data_choice)

            start_date, end_date = create_dates(
                num_days_range, adj_start_date_limit, end_date_limit
            )

            real_ohlc = RealOHLC(data_choice)
            dataframes = real_ohlc.create_df()
            dataframes = real_ohlc.real_case(dataframes, start_date, end_date)
            dataframes = normalize_ohlc_data(dataframes)
            answers[i] = f"Real: {start_date} to {end_date} {data_choice}"
        else:
            # how to show dates???
            random_ohlc = RandomOHLC(
                num_days_range, start_price, asset_name, volatility
            )
            random_ohlc.create_df()
            random_ohlc.create_realistic_ohlc()
            random_ohlc.normalize_ohlc_data()
            random_ohlc.resample_timeframes()
            # random_ohlc.drop_dates()
            half_dataframes = create_half_dataframes(random_ohlc.resampled_data)
            dataframes = random_ohlc.resampled_data
            answers[i] = f"Fake"

        for df in half_dataframes.values():
            fig = create_figure(df)
            fig.write_html(f"html/HABC-USD_{i}.html")
            fig.show(config=get_config())
            app.layout = app_update_layout(fig)

        # This is the full graph that only the admin should be able to see!
        ####################################################################
        # for timeframe, df in dataframes.items():
        #     fig = create_figure(df)
        #     fig.write_html(f"html/FABC-USD_{i}.html")
        ####################################################################

    pprint(answers)
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
