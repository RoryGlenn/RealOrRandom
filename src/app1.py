from time import sleep
from typing import Tuple
from pprint import pprint

import numpy as np
import pandas as pd
from dash import Dash
from faker import Faker
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import Dash, Input, Output, State, html, dcc, ctx

import cufflinks as cf
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

user_answers = {}
results = {}
current_graph_id = 0
TOTAL_GRAPHS = 1

GRAPH_IDS = [str(i).zfill(2) for i in range(20)]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "3D", "W", "M"]


# external_stylesheets = "https://codepen.io/chriddyp/pen/bWLwgP.css"
# app = Dash(__name__, external_stylesheets=[external_stylesheets])

# Initialize app
app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "RealorRandom"
server = app.server


# App layout
app.layout = html.Div(
    id="root",
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(
            id="header",
            children=[
                html.A(
                    html.Button("Enterprise Demo", className="link-button"),
                    href="https://plotly.com/get-demo/",
                ),
                html.A(
                    html.Button("Source Code", className="link-button"),
                    href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-opioid-epidemic",
                ),
                html.H4(children="Real or Random"),
                html.P(
                    id="description",
                    children=[
                        "Directions:",
                        html.P(
                            id="directions",
                            children=[
                                html.Div(
                                    [
                                        html.P(),
                                        html.P(
                                            "You are given a candle stick chart to analyze and make a future prediction with."
                                        ),
                                        html.P(
                                            "You are able to switch between any timeframe as much as you would like."
                                        ),
                                        html.P(
                                            "You are able to use the straight line drawing tool and the open freeform drawing tool as much as you would like."
                                        ),
                                        html.P(
                                            "After your analysis is complete you must answer all questions to the right."
                                        ),
                                        html.P(
                                            "You must complete the previous step before moving onto the next graph."
                                        ),
                                    ],
                                    style={"margin-left": "30px"},
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            id="timeframe-container",
                            children=[
                                html.P(
                                    id="timeframe-text",
                                    children="Select a timeframe:",
                                ),
                                html.Div(
                                    id="timeframebutton-container",
                                    style={"color": "#7fafdf"},
                                    children=[
                                        html.Button(
                                            "1m",
                                            id="1m",
                                            value="1m",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "5m",
                                            id="5m",
                                            value="5m",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "15m",
                                            id="15m",
                                            value="15m",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "30m",
                                            id="30m",
                                            value="30m",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "1h",
                                            id="1h",
                                            value="1h",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "4h",
                                            id="4h",
                                            value="4h",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "1D",
                                            id="1D",
                                            value="1D",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "3D",
                                            id="3D",
                                            value="3D",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "W",
                                            id="W",
                                            value="W",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                        html.Button(
                                            "M",
                                            id="M",
                                            value="M",
                                            n_clicks=0,
                                            style={"color": "#7fafdf"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # left graph
                        ############################################
                        html.Div(
                            id="maingraph-container",
                            children=[
                                dcc.Graph(
                                    id="main-graph",
                                    figure=go.Figure(
                                        data=go.Candlestick(),
                                        layout={
                                            "plot_bgcolor": "rgb(37,46,63)",
                                            "paper_bgcolor": "rgb(37,46,63)",
                                            "title": "Graph 00",
                                            "xaxis_title": "Time",
                                            "yaxis_title": "Price",
                                            "font": dict(size=14),
                                        },
                                    ),
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
                                    style={"width": "120vh", "height": "75vh"},
                                ),
                            ],
                        ),
                        ############################################
                    ],
                ),
                # Questions to the right of the screen
                html.Div(
                    id="question-container",
                    children=[
                        html.P(
                            id="question-header",
                            children="Answer the following questions:",
                        ),
                        html.Div(
                            dbc.Row(
                                [
                                    # 1 day
                                    html.Div(
                                        [
                                            html.P(
                                                "What will the price be 1 day after the last candle bars close price?"
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Slider(
                                                        id="1daybounds-slider",
                                                        step=0.5,
                                                        marks={
                                                            -100: {
                                                                "label": "-100%",
                                                                "style": {
                                                                    "color": "#f50"
                                                                },
                                                            },
                                                            0: {
                                                                "label": "0%",
                                                                "style": {
                                                                    "color": "#FFFFFF"
                                                                },
                                                            },
                                                            100: {
                                                                "label": "+100%",
                                                                "style": {
                                                                    "color": "#77b0b1"
                                                                },
                                                            },
                                                        },
                                                        min=-100,
                                                        max=100,
                                                        value=0,  # default value initially chosen
                                                        dots=True,  # True, False - insert dots, only when step>1
                                                        disabled=False,  # True,False - disable handle
                                                        updatemode="mouseup",  # 'mouseup', 'drag' - update value method
                                                        included=True,  # True, False - highlight handle
                                                        vertical=False,  # True, False - vertical, horizontal slider
                                                        verticalHeight=900,  # hight of slider (pixels) when vertical=True
                                                        className="None",
                                                        tooltip={
                                                            "always_visible": False,  # show current slider values
                                                            "placement": "bottom",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "width": "50%",
                                                    "margin-bottom": "30px",
                                                },
                                            ),
                                        ],
                                    ),
                                    # 5 day
                                    html.Div(
                                        [
                                            html.P(
                                                "What will the price be 5 days after the last candle bars close price?"
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Slider(
                                                        id="5daybounds-slider",
                                                        step=0.5,
                                                        marks={
                                                            -100: {
                                                                "label": "-100%",
                                                                "style": {
                                                                    "color": "#f50"
                                                                },
                                                            },
                                                            0: {
                                                                "label": "0%",
                                                                "style": {
                                                                    "color": "#FFFFFF"
                                                                },
                                                            },
                                                            100: {
                                                                "label": "+100%",
                                                                "style": {
                                                                    "color": "#77b0b1"
                                                                },
                                                            },
                                                        },
                                                        min=-100,
                                                        max=100,
                                                        value=0,  # default value initially chosen
                                                        dots=True,  # True, False - insert dots, only when step>1
                                                        disabled=False,  # True,False - disable handle
                                                        updatemode="mouseup",  # 'mouseup', 'drag' - update value method
                                                        included=True,  # True, False - highlight handle
                                                        vertical=False,  # True, False - vertical, horizontal slider
                                                        verticalHeight=900,  # hight of slider (pixels) when vertical=True
                                                        className="None",
                                                        tooltip={
                                                            "always_visible": False,  # show current slider values
                                                            "placement": "bottom",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "width": "50%",
                                                    "margin-bottom": "30px",
                                                },
                                            ),
                                        ],
                                    ),
                                    # 10 day
                                    html.Div(
                                        [
                                            html.P(
                                                "What will the price be 10 days after the last candle bars close price?"
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Slider(
                                                        id="10daybounds-slider",
                                                        step=0.5,
                                                        marks={
                                                            -100: {
                                                                "label": "-100%",
                                                                "style": {
                                                                    "color": "#f50"
                                                                },
                                                            },
                                                            0: {
                                                                "label": "0%",
                                                                "style": {
                                                                    "color": "#FFFFFF"
                                                                },
                                                            },
                                                            100: {
                                                                "label": "+100%",
                                                                "style": {
                                                                    "color": "#77b0b1"
                                                                },
                                                            },
                                                        },
                                                        min=-100,
                                                        max=100,
                                                        value=0,  # default value initially chosen
                                                        dots=True,  # True, False - insert dots, only when step>1
                                                        disabled=False,  # True,False - disable handle
                                                        updatemode="mouseup",  # 'mouseup', 'drag' - update value method
                                                        included=True,  # True, False - highlight handle
                                                        vertical=False,  # True, False - vertical, horizontal slider
                                                        verticalHeight=900,  # hight of slider (pixels) when vertical=True
                                                        className="None",
                                                        tooltip={
                                                            "always_visible": False,  # show current slider values
                                                            "placement": "bottom",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "width": "50%",
                                                    "margin-bottom": "30px",
                                                },
                                            ),
                                        ],
                                    ),
                                    # 30 day
                                    html.Div(
                                        [
                                            html.P(
                                                "What will the price be 30 days after the last candle bars close price?"
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Slider(
                                                        id="30daybounds-slider",
                                                        step=0.5,
                                                        marks={
                                                            -100: {
                                                                "label": "-100%",
                                                                "style": {
                                                                    "color": "#f50"
                                                                },
                                                            },
                                                            0: {
                                                                "label": "0%",
                                                                "style": {
                                                                    "color": "#FFFFFF"
                                                                },
                                                            },
                                                            100: {
                                                                "label": "+100%",
                                                                "style": {
                                                                    "color": "#77b0b1"
                                                                },
                                                            },
                                                        },
                                                        min=-100,
                                                        max=100,
                                                        value=0,
                                                        dots=True,
                                                        disabled=False,
                                                        updatemode="mouseup",
                                                        included=True,
                                                        vertical=False,
                                                        verticalHeight=900,
                                                        className="None",
                                                        tooltip={
                                                            "always_visible": False,
                                                            "placement": "bottom",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "width": "50%",
                                                    "margin-bottom": "30px",
                                                },
                                            ),
                                        ],
                                    ),
                                    # 60 day
                                    html.Div(
                                        [
                                            html.P(
                                                "What will the price be 60 days after the last candle bars close price?"
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Slider(
                                                        id="60daybounds-slider",
                                                        step=0.5,
                                                        marks={
                                                            -100: {
                                                                "label": "-100%",
                                                                "style": {
                                                                    "color": "#f50"
                                                                },
                                                            },
                                                            0: {
                                                                "label": "0%",
                                                                "style": {
                                                                    "color": "#FFFFFF"
                                                                },
                                                            },
                                                            100: {
                                                                "label": "+100%",
                                                                "style": {
                                                                    "color": "#77b0b1"
                                                                },
                                                            },
                                                        },
                                                        min=-100,
                                                        max=100,
                                                        value=0,  # default value initially chosen
                                                        dots=True,  # True, False - insert dots, only when step>1
                                                        disabled=False,  # True,False - disable handle
                                                        updatemode="mouseup",  # 'mouseup', 'drag' - update value method
                                                        included=True,  # True, False - highlight handle
                                                        vertical=False,  # True, False - vertical, horizontal slider
                                                        verticalHeight=900,  # hight of slider (pixels) when vertical=True
                                                        className="None",
                                                        tooltip={
                                                            "always_visible": False,  # show current slider values
                                                            "placement": "bottom",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "width": "50%",
                                                    "margin-bottom": "30px",
                                                },
                                            ),
                                        ],
                                    ),
                                    # Real or Random
                                    html.Div(
                                        [
                                            html.P("Is this graph real or random?"),
                                            dcc.Dropdown(
                                                id="realorrandom-dropdown",
                                                options=["Real", "Random"],
                                                value="",
                                            ),
                                        ],
                                        style={"width": "75%", "margin-bottom": "30px"},
                                    ),
                                    # Candle stick pattern
                                    html.Div(
                                        [
                                            html.P("What pattern do you see?"),
                                            dcc.Dropdown(
                                                options=[
                                                    # Else...
                                                    "No Pattern",
                                                    "I Don't Know",
                                                    # Bullish
                                                    "Hammer",
                                                    "Piercing Pattern",
                                                    "Bullish Engulfing",
                                                    "The Morning Star",
                                                    "Three White Soldiers",
                                                    "White Marubozu",
                                                    "Three Inside Up",
                                                    "Bullish Harami",
                                                    "Tweezer Bottom",
                                                    "Inverted Hammer",
                                                    "Three Outside Up",
                                                    "On-Neck Pattern",
                                                    "Bullish Counterattack",
                                                    # Bearish
                                                    "Hanging man",
                                                    "Dark cloud cover",
                                                    "Bearish Engulfing",
                                                    "The Evening Star",
                                                    "Three Black Crows",
                                                    "Black Marubozu",
                                                    "Three Inside Down",
                                                    "Bearish Harami",
                                                    "Shooting Star",
                                                    "Tweezer Top",
                                                    "Three Outside Down",
                                                    "Bearish Counterattack",
                                                    # Continuation Candlestick Patterns
                                                    "Doji",
                                                    "Spinning Top",
                                                    "Falling Three Methods",
                                                    "Rising Three Methods",
                                                    "Upside Tasuki Gap",
                                                    "Downside Tasuki Gap",
                                                    "Mat-Hold",
                                                    "Rising Window",
                                                    "Falling Window",
                                                    "High Wave",
                                                    # N/A
                                                    "Other",
                                                ],
                                                id="pattern-dropdown",
                                            ),
                                        ],
                                        style={"width": "75%", "margin-bottom": "30px"},
                                    ),
                                    # Confidence slider
                                    html.Div(
                                        [
                                            html.P(
                                                "What is your overall confidence in your answers?"
                                            ),
                                            html.P(
                                                "(0: not at all confident, 10: extremely confident)"
                                            ),
                                            dcc.Slider(
                                                id="confidence-slider",
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=5,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        style={"width": "75%", "margin-bottom": "30px"},
                                    ),
                                    # Submit button
                                    html.Div(
                                        [
                                            dcc.ConfirmDialogProvider(
                                                html.Th(
                                                    html.Button(
                                                        id="submit-button",
                                                        children="Submit",
                                                        type="button",
                                                        style={
                                                            "color": "rgb(44,254,193)"
                                                        },
                                                    )
                                                ),
                                                id="submit-provider",
                                                message="You will not be able to go back after submitting.\nAre you sure you want to continue?",
                                            ),
                                            html.Div(id="submit_output-provider"),
                                        ],
                                        style={"width": "25%", "margin-bottom": "10px"},
                                    ),
                                ],
                                style={
                                    "margin-left": "5%",
                                },
                            ),
                        ),
                    ],
                ),
            ],
            style={"width": "170vh", "height": "85vh"},
        ),
    ],
)


# @app.callback(Output("maingraph-title", "children"), [Input("timeframe-button", "value")])
def update_map_title(id: int):
    return "Graph ID: {0}".format(id)


# connect this call back with the new buttons placed inside the graph
@app.callback(
    Output("main-graph", "figure"),
    [Input(i, "n_clicks") for i in TIMEFRAMES],
)
def display_selected_timeframe(*args) -> go.Figure:
    btn_value = "1D" if not any(args) else ctx.triggered_id

    df = FrontEnd.half_dataframes[FrontEnd.timeframe_map[btn_value]]

    fig = go.Figure(
        data=go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
        ),
        layout={
            "plot_bgcolor": "rgb(37,46,63)",
            "paper_bgcolor": "rgb(37,46,63)",
            "title": "Graph 00",  # get the current graph id and add one ?
            "xaxis_title": "Time",
            "yaxis_title": "Price",
            "font": {"size": 14, "color": "#7fafdf"},
        },
    )

    # fig.update_yaxes(
    #     showspikes=True,
    #     spikemode="across",
    #     spikesnap="cursor",
    #     spikethickness=0.5,
    # )

    # fig.update_xaxes(
    #     showspikes=True,
    #     spikemode="across",
    #     spikesnap="cursor",
    #     spikethickness=0.5,

    # )

    # fig.update_layout(hoverdistance=0)
    # fig.update_traces(xaxis="x")
    # fig.update_traces(yaxis="y")

    buttons = [
        dict(label=i, method="update", args=[{"visible": [False]}]) for i in TIMEFRAMES
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57,
                y=1.2,
                buttons=list(buttons),
            )
        ]
    ),

    return fig


#########################################################################################################################################################################


# # add a click to the appropriate store.
# @app.callback(
#     Output("submit-button", "value"),
#     Input("submit-button", "n_clicks"),
#     State("1daybounds-slider", "value"),
#     State("5daybounds-slider", "value"),
#     State("10daybounds-slider", "value"),
#     State("30daybounds-slider", "value"),
#     State("60daybounds-slider", "value"),
#     State("realorrandom-dropdown", "value"),
#     State("pattern-dropdown", "value"),
#     State("confidence-slider", "value"),
# )
# def on_submit(
#     num_clicks: int,
#     daybounds1: list[float],
#     daybounds5: list[float],
#     daybounds10: list[float],
#     daybounds30: list[float],
#     daybounds60: list[float],
#     real_or_random: str,
#     pattern: str,
#     confidence: int,
# ):
#     global user_answers
#     global graph_id
#     global total_graphs

#     if num_clicks is None:
#         # prevent the None callbacks is important with the store component.
#         # you don't want to update the store for nothing.
#         raise PreventUpdate

#     user_answers[graph_id] = {
#         "1daybounds-slider": daybounds1,
#         "5daybounds-slider": daybounds5,
#         "10daybounds-slider": daybounds10,
#         "30daybounds-slider": daybounds30,
#         "60daybounds-slider": daybounds60,
#         "realorrandom-dropdown": real_or_random,
#         "pattern-dropdown": pattern,
#         "confidence-slider": confidence,
#     }

#     # if len(user_answers) == total_graphs:
#     #     calculate_results()

#     graph_id += 1
#     return user_answers


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

    real_ohlc = RealOHLC(
        DATA_PATH + "/" + data_choice,
        num_days,
        data_files=Download.get_data_filenames(DATA_FILENAMES),
    )

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
    global current_graph_id
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


def main() -> None:
    Faker.seed(np.random.randint(0, 10_000))
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

    for i in range(TOTAL_GRAPHS):
        FrontEnd.dataframes, FrontEnd.half_dataframes, answers["graph_" + str(i)] = (
            real_case(num_days, fake)
            if np.random.choice([False])
            else random_case(num_days, fake)
        )
        reset_indices(FrontEnd.dataframes, FrontEnd.half_dataframes)

    # calculate_results()
    pprint(answers)
    print()


if __name__ == "__main__":
    main()
    app.run_server(debug=True, port=8080)
