from typing import Tuple
from time import perf_counter
from datetime import datetime
from urllib import request

import numpy as np
import pandas as pd
from dash import Dash, html, dcc
from faker import Faker
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from dates import Dates
from frontend import FrontEnd
from real_ohlc import RealOHLC
from random_ohlc import RandomOHLC
from constants.constants import *

# creates the Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# creates the layout of the App
app.layout = html.Div(
    [
        html.H1("Real Time Charts"),
        dbc.Row(
            [
                dbc.Col(),
            ]
        ),
        html.Hr(),
        dcc.Interval(id="update", interval=5000),
        html.Div(id="page-content"),
    ],
    style={"margin-left": "5%", "margin-right": "5%", "margin-top": "20px"},
)

current_graph = None

timeframe_map = {
    "1_Minute": "1min",
    "5_Minute": "5min",
    "15_Minute": "15min",
    "30_Minute": "30min",
    "1_Hour": "1H",
    "2_Hour": "2H",
    "4_Hour": "4H",
    "1_Day": "1D",
    "1_Week": "1W",
    "1_Month": "1M",
}


@app.callback(
    Output("page-content", "figure"),
    # Input("update", "n_intervals"),
    State("timeframe-dropdown", "value"),
)
def update_ohlc_chart(user_timeframe: str):
    """A callback function that updates the graph every
    time a new timeframe is selected by the user"""
    global current_graph

    internal_timeframe = FrontEnd.timeframe_map[user_timeframe]
    df = FrontEnd.half_dataframes[internal_timeframe]

    fig = go.Figure(
        data=go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
    )

    return FrontEnd.get_graph_layout(current_graph)
    # app.layout = FrontEnd.get_graph_layout(fig)


def has_data(self) -> bool:
    return False

def download_data() -> None:
    import os
    import requests
    import sys

    data_url = 'https://github.com/RoryGlenn/RealOrRandom/tree/main/data'
    data_repo = 'data'
    
    # if os.path.exists(data_repo):
    #     # double check that we have all .csv files
    #     return
    
    response = requests.get(data_url)
    if response.ok:
        print("downloading csv data")
    else:
        print("could not download csv data")
        sys.exit(1)


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
    start_time = perf_counter()

    global current_graph

    download_data()

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

        FrontEnd.dataframes = dataframes
        FrontEnd.half_dataframes = half_dataframes

        current_graph = FrontEnd.create_figure(FrontEnd.half_dataframes["1D"], "1_Day")
        # app.layout = FrontEnd.app_create_layout(fig)

    print("Finished")
    app.run_server(debug=True)


if __name__ == "__main__":
    # from os import system
    # system("cls")
    main()
