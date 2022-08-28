from typing import Tuple

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
from constants.constants import DATA_PATH, SECONDS_IN_1DAY


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

def get_data_filenames(data_path: str) -> list[str]: 
    with open(data_path, 'r') as file:
        filenames = file.readlines()
    return [f.replace('\n', '') for f in filenames]    

def download_data(url: str, files_to_download: list[str], download_path: str) -> None:
    from requests import get
    from os import path, mkdir, listdir
    
    # create data directory if it doesn't exist
    if not path.exists(download_path):
        mkdir(download_path)
        
    for file in files_to_download:
        # if file is not in data folder, download it
        if file not in listdir(download_path):
            response = get(url, stream=True)
            with open(file, "wb") as fd:
                for chunk in response.iter_content(chunk_size=256):
                    fd.write(chunk)
            print(f"File {file} downloaded succesfully")


def create_half_dataframes(
    dataframes: dict[str, pd.DataFrame], exclusions=[]
) -> dict[str, pd.DataFrame]:
    """Creates a new dict that contains only the first half the data in the dataframes"""
    return {
        timeframe: df.iloc[: len(df) // 2]
        for timeframe, df in dataframes.items()
        if timeframe not in exclusions
    }


def real_case(
    num_days: int, fake: Faker, exclusions: list[str] = []
) -> Tuple[dict, dict, str]:
    """Creates the real case scenario"""

    # get the files and randomly decide which data to use
    files = Dates.get_filenames(DATA_PATH)
    data_choice = np.random.choice(files)

    real_ohlc = RealOHLC(DATA_PATH + "/" + data_choice, num_days)
    
    date_ranges = real_ohlc.get_start_end_dates()[data_choice]
    start_date_str, end_date_str = real_ohlc.get_start_end_date_strs(
        date_ranges, num_days
    )

    real_ohlc.create_df(start_date_str, end_date_str, merge_csvs=True)
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
    return (
        random_ohlc.resampled_data,
        half_dataframes,
        f"Random: {fake.name()}, Start Date: None, End Date: None",
    )


def main() -> None:
    timeframe_exclusions = ["1min", "5min", "15min", "30min", "1H", "2H", "4H"]
    url = "https://github.com/RoryGlenn/RealOrRandom/raw/main/data/"
    filenames = get_data_filenames('data_filenames.txt')
    download_path = 'data'
    download_data(url, filenames, download_path)

    Faker.seed(0)
    fake = Faker()
    total_graphs = 1
    num_days = 120  # 120 will be the standard
    answers = {}

    print("Starting test...")

    for i in range(total_graphs):
        dataframes, half_dataframes, answers[i] = (
            real_case(num_days, fake)
            if np.random.choice([True])
            else random_case(num_days, fake)
        )

        for timeframe in dataframes:
            dataframes[timeframe].reset_index(inplace=True)
        for timeframe in half_dataframes:
            half_dataframes[timeframe].reset_index(inplace=True)

        FrontEnd.dataframes = dataframes
        FrontEnd.half_dataframes = half_dataframes

    print("Finished")
    print("Answers:", answers)
    print()
    app.run_server()


if __name__ == "__main__":
    main()
