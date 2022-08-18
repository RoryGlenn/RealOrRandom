from datetime import datetime, timedelta

from time import perf_counter
from typing import Tuple

import pandas as pd
import numpy as np
from faker import Faker
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import plotly.graph_objects as go

from frontend import FrontEnd
from real_ohlc import RealOHLC
from random_ohlc import RandomOHLC
from constants.constants import *

# creates the Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


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


def get_date_limits(days: int, data_choice: int) -> Tuple[str, str]:
    """Returns the absolute start and end date for a specific data file"""
    d_range = get_data_date_ranges().get(data_choice)
    start_date__limit_l = [int(i) for i in d_range["start_date"].split("-")]

    # adjust the start_date 91 days after the original start date
    start_dt_limit = datetime(
        year=start_date__limit_l[0],
        month=start_date__limit_l[1],
        day=start_date__limit_l[2],
    )
    adjusted_start_dt = start_dt_limit + timedelta(days=days)
    adj_start_date_limit = adjusted_start_dt.strftime("%Y-%m-%d")
    end_date_limit = d_range["end_date"]
    return adj_start_date_limit, end_date_limit


def create_dates(
    num_days_range: int, start_date_limit: str, end_date_limit: str
) -> tuple[str, str]:
    """Randomly pick a start and end date within the given starting and ending bounds"""

    start_date_limit_l = [int(i) for i in start_date_limit.split("-")]
    end_date_limit_l = [int(i) for i in end_date_limit.split("-")]

    start_limit_dt = datetime(
        year=start_date_limit_l[0],
        month=start_date_limit_l[1],
        day=start_date_limit_l[2],
    )

    end_limit_dt = datetime(
        year=end_date_limit_l[0], month=end_date_limit_l[1], day=end_date_limit_l[2]
    )

    # get the number of days from the start date to the end date
    date_range_limit = end_limit_dt - start_limit_dt

    # create a list of all the dates within the given date bounds
    dt_list = [start_limit_dt + timedelta(days=x) for x in range(date_range_limit.days)]

    # pick a random day to start minus the given range
    start_i = np.random.randint(0, len(dt_list) - num_days_range)
    end_i = start_i + num_days_range

    start_random_dt = dt_list[start_i]
    end_random_dt = dt_list[end_i]
    return start_random_dt.strftime("%Y-%m-%d"), end_random_dt.strftime("%Y-%m-%d")


def create_half_dataframes(
    dataframes: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Creates a new dict that contains only the first half the data in the dataframes"""
    exclusions = ["1min", "5min", "15min", "30min", "1H", "2H", "4H"]
    return {
        timeframe: df.iloc[: len(df) // 2]
        for timeframe, df in dataframes.items()
        if timeframe not in exclusions
    }


def main() -> None:
    # frontend = FrontEnd()
    start_time = perf_counter()
    Faker.seed(0)
    fake = Faker()
    total_graphs = 1
    num_days = 120  # 120 will be the standard
    answers = {}
    exclusions = ["1min", "5min", "15min", "30min", "1H", "2H", "4H"]
    # app.layout = FrontEnd.app_create_layout()

    print("Starting test...")

    for i in range(total_graphs):
        dataframes = None
        half_dataframes = {}

        if np.random.randint(0, 1):
            # if False:
            days = 91
            data_choice = np.random.choice(list(get_data_date_ranges().keys()))

            adj_start_date_limit, end_date_limit = get_date_limits(days, data_choice)

            start_date_str, end_date_str = create_dates(
                num_days, adj_start_date_limit, end_date_limit
            )

            start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            num_days = end_date_dt - start_date_dt

            real_ohlc = RealOHLC(data_choice, num_days.days)
            real_ohlc.create_df(start_date_str, end_date_str)
            real_ohlc.normalize_ohlc_data()
            real_ohlc.resample_timeframes()

            answers[i] = f"Real: {start_date_str} to {end_date_str} {data_choice}"

            dataframes = real_ohlc.resampled_data
            half_dataframes = create_half_dataframes(real_ohlc.resampled_data)
        else:
            random_ohlc = RandomOHLC(
                total_days=num_days,
                start_price=100_000,
                name=fake.name(),
                volatility=np.random.uniform(1, 2),
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

            dataframes = random_ohlc.resampled_data
            half_dataframes = create_half_dataframes(random_ohlc.resampled_data)
            answers[i] = f"Fake"

        # loop bottle necks!!!
        for timeframe, df in half_dataframes.items():
            if timeframe in exclusions:
                continue

            fig = FrontEnd.create_figure(df, timeframe)
            # fig.write_html(f"html/HABC-USD_{timeframe}_{i}.html", config=get_config())
            app.layout = FrontEnd.app_create_layout(fig)

        # This is the full graph that only the admin should be able to see!
        ####################################################################
        # for timeframe, df in dataframes.items():
        #     fig = create_figure(df)
        #     fig.write_html(f"html/FABC-USD_{i}.html")
        ####################################################################

    print(answers)

    # any change made to this file will cause the server to recompile
    app.run_server(debug=True)

    time_elapsed = RandomOHLC.get_time_elapsed(start_time)
    print(f"Total time elapsed: {time_elapsed}")


if __name__ == "__main__":
    # from os import system
    # system("cls")
    main()
