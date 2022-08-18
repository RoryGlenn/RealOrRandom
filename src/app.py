from datetime import datetime
from time import perf_counter
from typing import Tuple

import pandas as pd
import numpy as np
from faker import Faker
from dash import Dash
import dash_bootstrap_components as dbc

from dates import Dates
from frontend import FrontEnd
from real_ohlc import RealOHLC
from random_ohlc import RandomOHLC
from constants.constants import *

# creates the Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def create_half_dataframes(
    dataframes: dict[str, pd.DataFrame], exclusions=[]
) -> dict[str, pd.DataFrame]:
    """Creates a new dict that contains only the first half the data in the dataframes"""
    return {
        timeframe: df.iloc[: len(df) // 2]
        for timeframe, df in dataframes.items()
        if timeframe not in exclusions
    }


def real_case(exclusions: list[str]) -> Tuple[dict, dict, str]:
    data_choice = np.random.choice(list(Dates.get_data_date_ranges().keys()))

    adj_start_date_limit, end_date_limit = Dates.get_date_limits(
        days=90 + 1, data_choice=data_choice
    )

    start_date_str, end_date_str = Dates.create_dates(
        num_days, adj_start_date_limit, end_date_limit
    )

    start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    num_days = end_date_dt - start_date_dt

    real_ohlc = RealOHLC(data_choice, num_days.days)
    real_ohlc.create_df(start_date_str, end_date_str)
    real_ohlc.normalize_ohlc_data()
    real_ohlc.resample_timeframes()

    half_dataframes = create_half_dataframes(real_ohlc.resampled_data, exclusions)
    answer = f"Real: {start_date_str} to {end_date_str} {data_choice}"
    return real_ohlc.resampled_data, half_dataframes, answer


def random_case(
    num_days: int, fake: Faker, exclusions: list[str]
) -> Tuple[dict, dict, str]:
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

    half_dataframes = create_half_dataframes(random_ohlc.resampled_data)
    return random_ohlc.resampled_data, half_dataframes, "Fake"


def main() -> None:
    start_time = perf_counter()

    Faker.seed(0)
    fake = Faker()
    total_graphs = 0
    num_days = 120  # 120 will be the standard
    answers = {}
    exclusions = ["1min", "5min", "15min", "30min", "1H", "2H", "4H"]

    print("Starting test...")

    for i in range(total_graphs):
        dataframes = None
        half_dataframes = {}

        if np.random.randint(0, 1):
            dataframes, half_dataframes, answers[i] = real_case(exclusions)
        else:
            dataframes, half_dataframes, answers[i] = random_case(
                num_days, fake, exclusions
            )

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
        #     fig = FrontEnd.create_figure(df)
        #     fig.write_html(f"html/FABC-USD_{i}.html")
        ####################################################################

    print(answers)

    # any change made to this file will cause the server to recompile
    app.run_server(debug=True)

    time_elapsed = RandomOHLC.get_time_elapsed(start_time)
    print(f"Total time elapsed: {time_elapsed}")


if __name__ == "__main__":
    from os import system

    system("cls")
    main()
