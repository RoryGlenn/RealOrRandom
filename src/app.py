from typing import Tuple
from time import perf_counter
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker
from dash import Dash, html, dcc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from dates import Dates
from frontend import FrontEnd
from real_ohlc import RealOHLC
from random_ohlc import RandomOHLC
from constants.constants import SECONDS_IN_1DAY


# creates the Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = FrontEnd.get_app_layout()


@app.callback(
    Output("page-content", "children"),
    Input("timeframe-dropdown", "value"),
)
def update_ohlc_chart(user_timeframe: str):
    """A callback function that updates the graph every
    time a new timeframe is selected by the user"""

    print("user_timeframe", user_timeframe)

    if user_timeframe is None:
        user_timeframe = "1_Day"

    df = half_dataframes[FrontEnd.timeframe_map[user_timeframe]]

    fig = go.Figure(
        data=go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
    )

    return FrontEnd.get_graph_layout(fig)


def download_and_unzip(url, extract_to):
    from io import BytesIO
    from zipfile import ZipFile
    import wget
    import os
    import sys

    if not os.path.exists(extract_to):
        os.mkdir(extract_to)

    response = wget.download(url, extract_to)
    if not os.path.exists(response):
        print(f"Could not download from {url}")
        sys.exit(1)

    try:
        b = BytesIO(response)
        zipfile = ZipFile(b)

        zipfile.extractall(path=extract_to)
    except Exception as e:
        print(e)


def create_half_dataframes(
    dataframes: dict[str, pd.DataFrame], exclusions=[]
) -> dict[str, pd.DataFrame]:
    """Creates a new dict that contains only the first half the data in the dataframes"""
    return {
        timeframe: df.iloc[: len(df) // 2]
        for timeframe, df in dataframes.items()
        if timeframe not in exclusions
    }


def real_case(exclusions: list[str] = []) -> Tuple[dict, dict, str]:
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
    num_days: int, fake: Faker, exclusions: list[str] = []
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

    half_dataframes = create_half_dataframes(random_ohlc.resampled_data, exclusions)
    return random_ohlc.resampled_data, half_dataframes, "Fake"


def main() -> None:
    global current_graph
    global dataframes
    global half_dataframes
    global app

    start_time = perf_counter()
    # data_url = "https://github.com/RoryGlenn/RealOrRandom/blob/main/data.zip"
    # data_repo = "data"
    # download_and_unzip(data_url, data_repo)

    Faker.seed(0)
    fake = Faker()
    total_graphs = 1
    num_days = 120  # 120 will be the standard
    answers = {}
    exclusions = ["1min", "5min", "15min", "30min", "1H", "2H", "4H"]

    print("Starting test...")

    for i in range(total_graphs):
        dataframes, half_dataframes, answers[i] = (
            real_case() if np.random.randint(0, 1) else random_case(num_days, fake)
        )

        for timeframe in dataframes:
            dataframes[timeframe].reset_index(inplace=True)
        for timeframe in half_dataframes:
            half_dataframes[timeframe].reset_index(inplace=True)

        # dataframes = dataframes
        # half_dataframes = half_dataframes
        FrontEnd.dataframes = dataframes
        FrontEnd.half_dataframes

        # current_graph = FrontEnd.create_figure(half_dataframes["1D"], "1_Day")

    print("Finished")
    app.run_server()


if __name__ == "__main__":
    main()
