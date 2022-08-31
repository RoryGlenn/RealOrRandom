from typing import Tuple

import numpy as np
import pandas as pd
from dash import Dash
from faker import Faker
import plotly.graph_objects as go

from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

from dates import Dates
from download import Download
from frontend import FrontEnd
from real_ohlc import RealOHLC
from random_ohlc import RandomOHLC
from constants.constants import (
    DATA_PATH,
    DOWNLOAD_PATH,
    GITHUB_URL,
    SECONDS_IN_1DAY,
    DATA_FILENAMES,
)

external_stylesheets = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = Dash(__name__, external_stylesheets=[external_stylesheets])
app.layout = FrontEnd.get_app_layout()
user_answers = {}
results = {}
graph_id = 0
total_graphs = 1


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

    fig = go.Figure(
        data=go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
        )
    )
    return FrontEnd.get_graph_layout(fig)


# add a click to the appropriate store.
@app.callback(
    Output("submit-button", "value"),
    Input("submit-button", "n_clicks"),
    State("1daybounds-slider", "value"),
    State("5daybounds-slider", "value"),
    State("10daybounds-slider", "value"),
    State("30daybounds-slider", "value"),
    State("60daybounds-slider", "value"),
    State("realorrandom-dropdown", "value"),
    State("pattern-dropdown", "value"),
    State("confidence-slider", "value"),
)
def on_submit(
    num_clicks: int,
    daybounds1: list[float],
    daybounds5: list[float],
    daybounds10: list[float],
    daybounds30: list[float],
    daybounds60: list[float],
    real_or_random: str,
    pattern: str,
    confidence: int,
):
    global user_answers
    global graph_id
    global total_graphs

    if num_clicks is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    user_answers[graph_id] = {
        "1daybounds-slider": daybounds1,
        "5daybounds-slider": daybounds5,
        "10daybounds-slider": daybounds10,
        "30daybounds-slider": daybounds30,
        "60daybounds-slider": daybounds60,
        "realorrandom-dropdown": real_or_random,
        "pattern-dropdown": pattern,
        "confidence-slider": confidence,
    }

    if len(user_answers) == total_graphs:
        calculate_results()

    graph_id += 1
    return user_answers


def create_half_dataframes(
    dataframes: dict[str, pd.DataFrame], exclusions=[]
) -> dict[str, pd.DataFrame]:
    """Creates a new dict that contains only the first half the data in the dataframes"""
    return {
        timeframe: df.iloc[: len(df) // 2]
        for timeframe, df in dataframes.items()
        if timeframe not in exclusions
    }


def reset_indices(
    dataframes: dict[str, pd.DataFrame], half_dataframes: dict[str, pd.DataFrame]
) -> None:
    """Resets the index for every dataframe in dataframes and half_dataframes"""
    {
        df.reset_index(inplace=True): hdf.reset_index(inplace=True)
        for df, hdf in zip(dataframes.values(), half_dataframes.values())
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

    answer = {
        "Real_Or_Random": "Real",
        "Name": fake.name(),
        "Start_Date": start_date_str,
        "End_Date": end_date_str,
        "File": data_choice,
    }
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
    answer = {
        "Real_Or_Random": "Random",
        "Name": fake.name(),
        "Start_Date": "None",
        "End_Date": "None",
        "File": "None",
    }
    return random_ohlc.resampled_data, half_dataframes, answer


def get_relative_change(initial_value: float, final_value: float) -> float:
    """Returns the relative change.
    Formula = (x2 - x1) / x1"""
    return (final_value - initial_value) / initial_value


def get_results(users_answers: dict, relative_change: float, day_number: int) -> dict:
    return {
        f"relative_change_{day_number}day": relative_change,
        "user_1day": users_answers[f"{day_number}daybounds-slider"],
        "user_off_by_1day": abs(relative_change)
        - abs(users_answers[f"{day_number}daybounds-slider"]),
        "user_real_or_random": users_answers["realorrandom-dropdown"],
        "user_pattern": users_answers["pattern-dropdown"],
        "user_confidence": users_answers["confidence-slider"],
    }


def calculate_results() -> None:
    """Compare the users guessed price to the actual price in the full dataframe"""
    from pprint import pprint

    global user_answers
    global graph_id
    global results

    # need to iterate over all graphs!!!!
    # right now, this only iterates over 1 graph
    for g_id, usrs_answer in user_answers.items():
        initial_price = FrontEnd.half_dataframes["1D"].loc[59, "Close"]

        future_1day = FrontEnd.dataframes["1D"].loc[60, "Close"]
        future_5day = FrontEnd.dataframes["1D"].loc[64, "Close"]
        future_10day = FrontEnd.dataframes["1D"].loc[69, "Close"]
        future_30day = FrontEnd.dataframes["1D"].loc[89, "Close"]
        future_60day = FrontEnd.dataframes["1D"].loc[119, "Close"]

        relative_change_1day = get_relative_change(initial_price, future_1day) * 100
        relative_change_5day = get_relative_change(initial_price, future_5day) * 100
        relative_change_10day = get_relative_change(initial_price, future_10day) * 100
        relative_change_30day = get_relative_change(initial_price, future_30day) * 100
        relative_change_60day = get_relative_change(initial_price, future_60day) * 100

        results[g_id] = [
            get_results(usrs_answer, relative_change_1day, 1),
            get_results(usrs_answer, relative_change_5day, 5),
            get_results(usrs_answer, relative_change_10day, 10),
            get_results(usrs_answer, relative_change_30day, 30),
            get_results(usrs_answer, relative_change_60day, 60),
        ]

    pprint(results)


# TODO:
#     Save and continue button
#     Turn timeframe dropdown into trading view buttons instead
#     Loading bar
#     prevent the user from clicking the submit button until everything is filled out
#     Results page
#     Dataframes create 121 instead of 120 days (off by 1) (THE FIRST CANDLE BAR IS BROKEN!)
#     Email results?
def main() -> None:
    Faker.seed(np.random.randint(0, 10000))
    fake = Faker()
    answers = {}
    num_days = 120  # 120 will be the standard
    timeframe_exclusions = ["1min", "5min", "15min", "30min", "1H", "2H", "4H"]

    Download.download_data(
        url=GITHUB_URL,
        files_to_download=Download.get_data_filenames(DATA_FILENAMES),
        download_path=DOWNLOAD_PATH,
    )

    print("Starting test...")

    for i in range(total_graphs):
        FrontEnd.dataframes, FrontEnd.half_dataframes, answers["graph_" + str(i)] = (
            real_case(num_days, fake)
            if np.random.choice([True, False])
            else random_case(num_days, fake)
        )

        reset_indices(FrontEnd.dataframes, FrontEnd.half_dataframes)

    from pprint import pprint

    pprint(answers)
    print()

    app.run_server(debug=True, port=8080)


if __name__ == "__main__":
    main()
