from typing import Tuple
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from dash import Dash
from faker import Faker
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from dates import Dates
from frontend import FrontEnd
from real_ohlc import RealOHLC
from random_ohlc import RandomOHLC
from constants.constants import SECONDS_IN_1DAY

# redownload every .csv minute file and append all info into one dataframe.


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = FrontEnd.get_app_layout()


@app.callback(
    Output("page-content", "children"),
    Input("timeframe-dropdown", "value"),
)
def update_ohlc_chart(user_timeframe: str):
    """A callback function that updates the graph every
    time a new timeframe is selected by the user"""

    if user_timeframe is None:
        user_timeframe = "1_Day"

    df = FrontEnd.half_dataframes[FrontEnd.timeframe_map[user_timeframe]]
    date = "date" if "date" in df.columns else "Date"
    open = "open" if "open" in df.columns else "Open"
    high = "high" if "high" in df.columns else "High"
    low = "low" if "low" in df.columns else "Low"
    close = "close" if "close" in df.columns else "Close"

    fig = go.Figure(
        data=go.Candlestick(
            x=df[date],
            open=df[open],
            high=df[high],
            low=df[low],
            close=df[close],
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


def get_start_end_date_strs(date_ranges: dict, num_days: int) -> Tuple[str, str]:
    from datetime import timedelta

    # get the total number of days we can use
    start_date_dt = datetime.strptime(date_ranges["start_date"], "%Y-%m-%d %H:%M:%S")
    end_date_dt = datetime.strptime(date_ranges["end_date"], "%Y-%m-%d %H:%M:%S")

    # get the number of days from the start date to the end date
    diff_dt = end_date_dt - start_date_dt

    # Create a list of all the dates within the given date bounds.
    # Limit the total number of days we can use to diff_dt.days-num_days
    # so the last 'num_days' will be off limits to the start date.
    # By doing this, we protect ourselves from an out of bounds error
    dt_list = [
        start_date_dt + timedelta(days=x) for x in range(diff_dt.days - num_days)
    ]

    # randomly choose a start date, then go 'num_days' into the future to get the end date
    start_date_dt = np.random.choice(dt_list)
    end_date_dt = start_date_dt + timedelta(days=num_days)

    # create the start and end date strings
    start_date_str = start_date_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S")
    return start_date_str, end_date_str


def real_case(num_days: int, fake: Faker, exclusions: list[str] = []) -> Tuple[dict, dict, str]:
    """Creates the real case scenario"""
    # from datetime import timedelta

    # get the files and randomly decide which data to use
    files = Dates.get_filenames("data")
    data_choice = np.random.choice(files)
    date_ranges = Dates.get_start_end_dates()[data_choice]

    start_date_str, end_date_str = get_start_end_date_strs(date_ranges, num_days)

    real_ohlc = RealOHLC("data/" + data_choice, num_days)  # diff_dt.days)
    real_ohlc.create_df(start_date_str, end_date_str)
    real_ohlc.normalize_ohlc_data()

    real_ohlc.abstract_dates()
    real_ohlc.resample_timeframes()

    half_dataframes = create_half_dataframes(real_ohlc.resampled_data, exclusions)
    answer = f"Real -> Name: {fake.name()}, Start Date: {start_date_str}, End Date: {end_date_str}, File: {data_choice}"
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
    return random_ohlc.resampled_data, half_dataframes, f"Random: {fake.name()}, Start Date: None, End Date: None"


def main() -> None:
    start_time = perf_counter()
    data_url = "https://github.com/RoryGlenn/RealOrRandom/blob/main/data.zip"
    data_repo = "data"
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
            real_case(num_days, fake)
            if np.random.choice([True, False])
            else random_case(num_days, fake)
        )

        for timeframe in dataframes:
            dataframes[timeframe].reset_index(inplace=True)
        for timeframe in half_dataframes:
            half_dataframes[timeframe].reset_index(inplace=True)

        FrontEnd.dataframes = dataframes
        FrontEnd.half_dataframes = half_dataframes

    print("Finished")
    print("answers:", answers)
    print()
    app.run_server()


if __name__ == "__main__":
    main()
