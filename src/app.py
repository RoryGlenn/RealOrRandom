# import sys 
# from pprint import pprint
# sys.path.append(r'C:\Users\glenn\OneDrive\Desktop\RandomDataGenerator\venv\Scripts\python.exe')
# pprint(sys.path)


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


dataframes = {}
half_dataframes = {}
figures = {t: None for t, _ in timeframe_map.items()}


def create_figure(df: pd.DataFrame, graph_title: str) -> go.Figure:
    """Create the figure with the dataframe passed in"""
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


def get_timeframe_dropdown(timeframes: list[str]) -> html.Div:
    return html.Div(
        [
            html.P("Timeframe"),
            dcc.Dropdown(
                id="timeframe-dropdown",
                options=[
                    {"label": timeframe, "value": timeframe} for timeframe in timeframes
                ],
                value="1_Day",
            ),
        ]
    )


def get_graph_layout(fig: go.Figure) -> html.Div:
    """Updates the layout for the graph figure"""

    return html.Div(
        [
            dcc.Graph(
                id='graph-main',
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
                style={'width': '170vh', 'height': '90vh'}

            )
        ]
    )


@app.callback(
    Output("page-content", "children"),
    Input("timeframe-dropdown", "value"),
    State("timeframe-dropdown", "value"),
)
def update_ohlc_chart(intervals: int, user_timeframe: str):
    """A callback function that updates the graph every
    time a new timeframe is selected by the user"""
    global current_graph

    if user_timeframe is None:
        user_timeframe = "1_Day"

    df = half_dataframes[timeframe_map[user_timeframe]]

    fig = go.Figure(
        data=go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
    )

    return get_graph_layout(fig)


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

        dataframes = dataframes
        half_dataframes = half_dataframes

        current_graph = create_figure(half_dataframes["1D"], "1_Day")

    print("Finished")
    app.run_server()


app.layout = html.Div(
    [
        html.H1("Chart"),
        dbc.Row(
            [
                dbc.Col(get_timeframe_dropdown(list(timeframe_map.keys()))),
            ]
        ),
        dbc.Row(),
        html.Hr(),
        html.Div(id="page-content"),
        # dcc.Graph(id='my-graph',style={'width': '90vh', 'height': '90vh'}) 

    ],
    style={"margin-left": "5%", "margin-right": "5%", "margin-top": "20px"},
)


if __name__ == "__main__":
    main()
