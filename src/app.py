import random
import datetime
from datetime import timedelta

from time import perf_counter
from typing import Tuple

import numpy as np
import pandas as pd
from faker import Faker
from dash import Dash, dcc, html
import plotly.graph_objects as go

from real_ohlc import RealOHLC
from constants.constants import *
from random_ohlc import RandomOHLC


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


def create_figure(df: pd.DataFrame, graph_title: str) -> go.Figure:
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
        title=graph_title,
        xaxis_title="Date",
        yaxis_title="Price",
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
    return {
        timeframe: dataframes[timeframe].iloc[: len(dataframes[timeframe]) // 2]
        for timeframe in dataframes
    }


def main() -> None:
    start_time = perf_counter()
    Faker.seed(0)
    fake = Faker()
    total_graphs = 1
    num_days = 120  # 120 will be the standard
    answers = {}
    app = Dash()

    print("Starting test...")

    for i in range(total_graphs):
        dataframes = None
        half_dataframes = {}

        # if random.randint(0, 1):
        if True:
            days = 91
            data_choice = random.choice(list(get_data_date_ranges().keys()))

            adj_start_date_limit, end_date_limit = get_date_limits(days, data_choice)

            start_date_str, end_date_str = create_dates(
                num_days, adj_start_date_limit, end_date_limit
            )

            start_date_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
            num_days = end_date_dt - start_date_dt

            real_ohlc = RealOHLC(data_choice, num_days.days)
            real_ohlc.create_df(start_date_str, end_date_str)
            real_ohlc.normalize_ohlc_data()
            real_ohlc.resample_timeframes()

            answers[i] = f"Real: {start_date_str} to {end_date_str} {data_choice}"

            half_dataframes = create_half_dataframes(real_ohlc.resampled_data)
            dataframes = real_ohlc.resampled_data
        else:
            start_rohlc = perf_counter()
            random_ohlc = RandomOHLC(
                total_days=num_days,
                start_price=100_000,
                name=fake.name(),
                volatility=random.uniform(1, 2),
            )

            random_ohlc.generate_random_df(
                random_ohlc.total_days * SECONDS_IN_1DAY,
                "1S",
                random_ohlc.start_price,
                random_ohlc.volatility,
            )
            random_ohlc.create_realistic_ohlc()
            random_ohlc.normalize_ohlc_data()
            random_ohlc.resample_timeframes()

            half_dataframes = create_half_dataframes(random_ohlc.resampled_data)
            dataframes = random_ohlc.resampled_data
            answers[i] = f"Fake"

            print("Random OHLC elapsed: ", RandomOHLC.get_time_elapsed(start_rohlc))

        # loop bottle necks!!!
        for timeframe, df in half_dataframes.items():
            if timeframe in ["1min", "5min", "15min", "30min"]:
                continue

            fig = create_figure(df, timeframe)
            fig.write_html(f"html/HABC-USD_{timeframe}_{i}.html", config=get_config())
            fig.show(config=get_config())  # put me back in!
            # app.layout = app_update_layout(fig)

        # This is the full graph that only the admin should be able to see!
        ####################################################################
        # for timeframe, df in dataframes.items():
        #     fig = create_figure(df)
        #     fig.write_html(f"html/FABC-USD_{i}.html")
        ####################################################################

    # pprint(answers)

    # any change made to this file will cause the server to recompile
    # app.run_server(debug=True)

    time_elapsed = RandomOHLC.get_time_elapsed(start_time)
    print(f"Total time elapsed: {time_elapsed}")


if __name__ == "__main__":
    from os import system

    system("cls")
    main()


# to create drop down menu for timeframes
# https://www.youtube.com/watch?v=RwlqlGUDLkg
