from typing import Tuple
from pprint import pprint
from time import sleep

import numpy as np
import pandas as pd
from dash import Dash
from faker import Faker
import plotly.graph_objects as go

from dash import Dash, Input, Output, State, html, dcc, ctx
import dash_bootstrap_components as dbc
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
import cufflinks as cf

user_answers = {}
results = {}
graph_id = 0
total_graphs = 1

graph_ids = [str(i).zfill(2) for i in range(20)]


external_stylesheets = "https://codepen.io/chriddyp/pen/bWLwgP.css"
# app = Dash(__name__, external_stylesheets=[external_stylesheets])
# app.layout = FrontEnd.get_app_layout()

# Initialize app
app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "US Opioid Epidemic"
server = app.server

# Load data
import pathlib
import os

APP_PATH = str(pathlib.Path(__file__).parent.resolve())

df_lat_lon = pd.read_csv(os.path.join("", os.path.join("data", "lat_lon_counties.csv")))
df_lat_lon["FIPS "] = df_lat_lon["FIPS "].apply(lambda x: str(x).zfill(5))

df_full_data = pd.read_csv(
    os.path.join("", os.path.join("data", "age_adjusted_death_rate_no_quotes.csv"))
)
df_full_data["County Code"] = df_full_data["County Code"].apply(
    lambda x: str(x).zfill(5)
)
df_full_data["County"] = (
    df_full_data["Unnamed: 0"] + ", " + df_full_data.County.map(str)
)

YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "3D", "W", "M"]


BINS = [
    "0-2",
    "2.1-4",
    "4.1-6",
    "6.1-8",
    "8.1-10",
    "10.1-12",
    "12.1-14",
    "14.1-16",
    "16.1-18",
    "18.1-20",
    "20.1-22",
    "22.1-24",
    "24.1-26",
    "26.1-28",
    "28.1-30",
    ">30",
]

DEFAULT_COLORSCALE = [
    "#f2fffb",
    "#bbffeb",
    "#98ffe0",
    "#79ffd6",
    "#6df0c8",
    "#69e7c0",
    "#59dab2",
    "#45d0a5",
    "#31c194",
    "#2bb489",
    "#25a27b",
    "#1e906d",
    "#188463",
    "#157658",
    "#11684d",
    "#10523e",
]

DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"


# App layout
# Once page is loaded, every callback is triggered
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
                    children="† Deaths are classified using the International Classification of Diseases, \
                    Tenth Revision (ICD–10). Drug-poisoning deaths are defined as having ICD–10 underlying \
                    cause-of-death codes X40–X44 (unintentional), X60–X64 (suicide), X85 (homicide), or Y10–Y14 \
                    (undetermined intent).",
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
                                # dcc.Slider(
                                #     id="timeframe-button",
                                #     min=min(YEARS),
                                #     max=max(YEARS),
                                #     value=min(YEARS),
                                #     marks={
                                #         str(year): {
                                #             "label": str(year),
                                #             "style": {"color": "#7fafdf"},
                                #         }
                                #         for year in YEARS
                                #     },
                                # ),
                                html.Div(
                                    id="timeframebuttons-container",
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
                                html.P(
                                    "Graph ID: 00",
                                    id="maingraph-title",
                                ),
                                dcc.Graph(
                                    id="main-graph",
                                    figure=go.Figure(
                                        data=go.Candlestick(),
                                        layout={
                                            "plot_bgcolor": "rgb(37,46,63)",
                                            "paper_bgcolor": "rgb(37,46,63)",
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
                                    style={"width": "120vh", "height": "55vh"},
                                ),
                            ],
                        ),
                        ############################################
                    ],
                ),
                # chart to the right of the screen
                html.Div(
                    id="graph-container",
                    children=[
                        html.P(id="chart-selector", children="Select chart:"),
                        dcc.Dropdown(
                            options=[
                                {
                                    "label": "Histogram of total number of deaths (single year)",
                                    "value": "show_absolute_deaths_single_year",
                                },
                                {
                                    "label": "Histogram of total number of deaths (1999-2016)",
                                    "value": "absolute_deaths_all_time",
                                },
                                {
                                    "label": "Age-adjusted death rate (single year)",
                                    "value": "show_death_rate_single_year",
                                },
                                {
                                    "label": "Trends in age-adjusted death rate (1999-2016)",
                                    "value": "death_rate_all_time",
                                },
                            ],
                            value="show_death_rate_single_year",
                            id="chart-dropdown",
                        ),
                        dcc.Graph(
                            id="selected-timeframe2",
                            figure=dict(
                                data=[dict(x=0, y=0)],
                                layout=dict(
                                    paper_bgcolor="#F4F4F8",
                                    plot_bgcolor="#F4F4F8",
                                    autofill=True,
                                    margin=dict(t=75, r=50, b=100, l=50),
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


#         dcc.Store(id="submit"),
#         dbc.Row(
#             [
#                 # timeframe dropdown
#                 dbc.Col(FrontEnd.get_timeframe_dropdown()),

#             ]
#         ),
#         dbc.Row(),
#         html.Hr(),
#         html.Div(id="page-content"),
#         # Bounds dropdown
#         html.Div(
#             dbc.Row(
#                 [
#                     # 1 day
#                     FrontEnd.get_bounds_slider(
#                         text="What will the price be 1 day after the last candle bars close price?",
#                         id="1daybounds-slider",
#                     ),
#                 ]
#             ),
#         ),
#         html.Div(
#             dbc.Row(
#                 [
#                     # 5 days
#                     FrontEnd.get_bounds_slider(
#                         text="What will the price be 5 days after the last candle bars close price?",
#                         id="5daybounds-slider",
#                     ),
#                 ]
#             ),
#         ),
#         html.Div(
#             dbc.Row(
#                 [
#                     # 10 days
#                     FrontEnd.get_bounds_slider(
#                         text="What will the price be 10 days after the last candle bars close price?",
#                         id="10daybounds-slider",
#                     ),
#                 ]
#             ),
#         ),
#         html.Div(
#             dbc.Row(
#                 [
#                     # 30 days
#                     FrontEnd.get_bounds_slider(
#                         text="What will the price be 30 days after the last candle bars close price?",
#                         id="30daybounds-slider",
#                     ),
#                 ]
#             ),
#         ),
#         html.Div(
#             dbc.Row(
#                 [
#                     # 60 days
#                     FrontEnd.get_bounds_slider(
#                         text="What will the price be 60 days after the last candle bars close price?",
#                         id="60daybounds-slider",
#                     ),
#                 ],
#             ),
#         ),
#         # Real or Random
#         FrontEnd.get_real_or_random_dropdown(),
#         # pattern
#         FrontEnd.get_pattern_dropdown(),
#         # confidence
#         FrontEnd.get_confidence_slider(),
#         FrontEnd.submit_button(),
#     ],
#     style={
#         "margin-left": "5%",
#         "margin-right": "5%",
#         "margin-top": "20px",
#         "margin-bottom": "200px",
#     },
# )


# @app.callback(Output("maingraph-title", "children"), [Input("timeframe-button", "value")])
def update_map_title(id: int):
    return "Graph ID: {0}".format(id)


# ONLY the timeframe button will change the graph!!!
@app.callback(
    Output("main-graph", "figure"),
    [Input(i, "n_clicks") for i in TIMEFRAMES],
)
def display_selected_timeframe(*args) -> go.Figure:
    btn_value = "1D" if not any(args) else ctx.triggered_id

    df = FrontEnd.half_dataframes[FrontEnd.timeframe_map[btn_value]]

    return go.Figure(
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
        },
    )


# @app.callback(
#     Output("main-graph", "config"),
#     Input("timeframe-button", "value"),
# )
# def update_config(timeframe_button: str) -> dict:
#     print("update config")

#     while FrontEnd.half_dataframes is None:
#         print("sleeping till half dataframes")
#         sleep(1)

#     config = {
#         "doubleClickDelay": 1000,
#         "scrollZoom": True,
#         "displayModeBar": True,
#         "showTips": True,
#         "displaylogo": True,
#         "fillFrame": False,
#         "autosizable": True,
#         "modeBarButtonsToAdd": [
#             "drawline",
#             "drawopenpath",
#             "drawclosedpath",
#             "eraseshape",
#         ],
#     }
#     return config


#########################################################################################################################################################################
# @app.callback(
#     Output("main-graph", "children"),
#     Input("timeframe-button", "value"),
# )
# def update_ohlc_chart(user_timeframe: str):
#     """A callback function that updates the graph every
#     time a new timeframe is selected by the user"""

#     if user_timeframe is None:
#         user_timeframe = "1_Day"

#     df = FrontEnd.half_dataframes[FrontEnd.timeframe_map[user_timeframe]]

#     fig = go.Figure(
#         data=go.Candlestick(
#             x=df["Date"],
#             open=df["Open"],
#             high=df["High"],
#             low=df["Low"],
#             close=df["Close"],
#         )
#     )
#     return FrontEnd.get_graph_layout(fig)


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

    # if len(user_answers) == total_graphs:
    #     calculate_results()

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


# 8/31/22
# After the submit button is pressed, show the results page.


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

    for i in range(total_graphs):
        FrontEnd.dataframes, FrontEnd.half_dataframes, answers["graph_" + str(i)] = (
            real_case(num_days, fake)
            if np.random.choice([False])
            else random_case(num_days, fake)
        )
        reset_indices(FrontEnd.dataframes, FrontEnd.half_dataframes)

    # how do we pass the dataframes to the graph to be displayed???????????

    # calculate_results()
    pprint(answers)
    print()


if __name__ == "__main__":
    main()
    app.run_server(debug=False, port=8080)
