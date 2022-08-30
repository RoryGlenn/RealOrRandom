from typing import Tuple

import numpy as np
import pandas as pd
from dash import Dash
from faker import Faker
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# from dash.dependencies import Input, Output, State
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

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
app.layout = FrontEnd.get_app_layout()
results = {}
graph_id = 0


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
    Output("submit", "value"),
    Input("submit-button", "n_clicks"),
    State("1dayupperbounds-dropdown", "value"),
    State("1daylowerbounds-dropdown", "value"),
    State("5dayupperbounds-dropdown", "value"),
    State("5daylowerbounds-dropdown", "value"),
    State("10dayupperbounds-dropdown", "value"),
    State("10daylowerbounds-dropdown", "value"),
    State("30dayupperbounds-dropdown", "value"),
    State("30daylowerbounds-dropdown", "value"),
    State("60dayupperbounds-dropdown", "value"),
    State("60daylowerbounds-dropdown", "value"),
    State("realorrandom-dropdown", "value"),
    State("pattern-textbox", "value"),
    State("confidence-slider", "value"),
)
def on_submit(
    n_clicks: int,
    dayupperbounds1: float,
    daylowerbounds1: float,
    dayupperbounds5: float,
    daylowerbounds5: float,
    dayupperbounds10: float,
    daylowerbounds10: float,
    dayupperbounds30: float,
    daylowerbounds30: float,
    dayupperbounds60: float,
    daylowerbounds60: float,
    real_or_random: str,
    pattern: str,
    confidence: int,
):
    global results
    global graph_id

    if n_clicks is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    results[graph_id] = {
        "1dayupperbounds-dropdown": dayupperbounds1,
        "1daylowerbounds-dropdown": daylowerbounds1,
        "5dayupperbounds-dropdown": dayupperbounds5,
        "5daylowerbounds-dropdown": daylowerbounds5,
        "10dayupperbounds-dropdown": dayupperbounds10,
        "10daylowerbounds-dropdown": daylowerbounds10,
        "30dayupperbounds-dropdown": dayupperbounds30,
        "30daylowerbounds-dropdown": daylowerbounds30,
        "60dayupperbounds-dropdown": dayupperbounds60,
        "60daylowerbounds-dropdown": daylowerbounds60,
        "realorrandom-dropdown": real_or_random,
        "pattern-textbox": pattern,
        "confidence-slider": confidence,
    }

    graph_id += 1
    return results


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


# TODO:
# Save and continue button
# Turn timeframe dropdown into trading view buttons instead
# Loading bar
# prevent the user from clicking the submit button until everything is filled out
# Results page
# Email results?
def main() -> None:
    Faker.seed(np.random.randint(0, 10000))
    fake = Faker()
    answers = {}
    num_days = 120  # 120 will be the standard
    total_graphs = 1
    timeframe_exclusions = ["1min", "5min", "15min", "30min", "1H", "2H", "4H"]

    Download.download_data(
        url=GITHUB_URL,
        files_to_download=Download.get_data_filenames(DATA_FILENAMES),
        download_path=DOWNLOAD_PATH,
    )

    print("Starting test...")

    for i in range(total_graphs):
        FrontEnd.dataframes, FrontEnd.half_dataframes, answers[i] = (
            real_case(num_days, fake)
            if np.random.choice([False])
            else random_case(num_days, fake)
        )

        reset_indices(FrontEnd.dataframes, FrontEnd.half_dataframes)

    print("Finished")
    print("Answers:", answers)
    print()
    app.run_server(debug=True, port=8080)


if __name__ == "__main__":
    main()
