from pprint import pprint
from sys import exit as sys_exit

from dash import Dash
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import Dash, Input, Output, State, html, dcc, ctx

from constants.constants import TIMEFRAME_MAP, TIMEFRAMES, TOTAL_GRAPHS
from case_handler import CaseHandler

from logr import Logr

# setup the logging
logr = Logr()
logger = logr.setup_logger("root")
logger.debug("Starting Real or Random...")

case_hand = CaseHandler(num_days=120)
case_hand.init()

# Initialize app
app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "RealorRandom"
server = app.server


#######################################################################
# App Layout
#######################################################################

app.layout = html.Div(
    id="root",
    children=[
        dcc.Location(id="dummyurl", refresh=False),
        html.Div(
            id="header",
            children=[
                html.A(
                    html.Button("Source Code", className="link-button"),
                    href="https://github.com/RoryGlenn/RealOrRandom",
                ),
                html.A(
                    html.Button("Refresh Data", className="link-button", id="refresh"),
                    href="/",
                ),
                html.H4(children="Real Or Random"),
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
                                    # style={"color": "#7fafdf", "width": "50%"},
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
                                            "title": "Graph 00",  # get the current graph id and add one ?
                                            "xaxis_title": "Time",
                                            "yaxis_title": "Price",
                                            "font": {"size": 14, "color": "#7fafdf"},
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
                                                                    "color": "#77b181"
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
                                                                    "color": "#77b181"
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
                                                                    "color": "#77b181"
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
                                                                    "color": "#77b181"
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
                                                                    "color": "#77b181"
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
                                    # Real or Random
                                    html.Div(
                                        [
                                            html.P("Is this graph real or random?"),
                                            dcc.Dropdown(
                                                id="realorrandom-dropdown",
                                                options=["Real", "Random"],
                                                value="Real",
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
                                                value="No Pattern",
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
                                            html.P("(0: very low, 10: very high)"),
                                            dcc.Slider(
                                                id="confidence-slider",
                                                min=1,
                                                max=10,
                                                step=1,
                                                value=1,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        style={"width": "75%", "margin-bottom": "30px"},
                                    ),
                                    html.Button(
                                        id="submit-button",
                                        children="Submit",
                                        type="button",
                                        className="link-button",
                                        n_clicks=0,
                                        style={
                                            "color": "rgb(44,254,193)",
                                            "margin-right": "75%",
                                            "margin-top": "5%",
                                        },
                                    ),
                                ],
                                style={"margin-left": "5%"},
                            ),
                        ),
                    ],
                ),
            ],
            style={"width": "170vh", "height": "85vh"},
        ),
        html.Div(id="display_selected_timeframe", children=[]),
        html.Div(id="update_map_title", children=[]),
    ],
)


#######################################################################
# CallBacks
#######################################################################


@app.callback(
    Output("update_map_title", "children"),
    Input("submit-confirm", "n_clicks"),
)
def update_map_title(*args):
    
    return "Graph ID: {0}".format(id)


@app.callback(
    Output("display_selected_timeframe", "children"),
    Input("dummyurl", "refresh"),
)
def generate_graph(*args, **kwargs) -> tuple[int, int]:
    global case_hand

    case_hand.real_case(
        num_days=case_hand.num_days
    ) if case_hand.choose() else case_hand.random_case(num_days=case_hand.num_days)
    case_hand.reset_indices()
    
    logger.debug(f"Created graph {case_hand.curr_graph_id}")
    logger.debug(case_hand.answer)

    fig = go.Figure(
        data=go.Candlestick(
            x=case_hand.half_dataframes["1D"]["date"],
            open=case_hand.half_dataframes["1D"]["open"],
            high=case_hand.half_dataframes["1D"]["high"],
            low=case_hand.half_dataframes["1D"]["low"],
            close=case_hand.half_dataframes["1D"]["close"],
        ),
        layout={
            "plot_bgcolor": "rgb(37,46,63)",
            "paper_bgcolor": "rgb(37,46,63)",
            "title": "Graph 00",  # get the current graph id
            "xaxis_title": "Time",
            "yaxis_title": "Price",
            "font": {"size": 14, "color": "#7fafdf"},
        },
    )
    return fig


@app.callback(
    Output("main-graph", "figure"),
    Input("display_selected_timeframe", "children"),
    [Input(i, "n_clicks") for i in TIMEFRAMES],
)
def display_selected_timeframe(children: dict, *buttons: tuple[int]) -> go.Figure:
    global case_hand

    if case_hand.half_dataframes is None:
        generate_graph()

    btn_value = (
        "1D"
        if not any(buttons) or ctx.triggered_id == "display_selected_timeframe"
        else ctx.triggered_id
    )
    df = case_hand.half_dataframes[TIMEFRAME_MAP[btn_value]]

    return go.Figure(
        data=go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        ),
        layout={
            "plot_bgcolor": "rgb(37,46,63)",
            "paper_bgcolor": "rgb(37,46,63)",
            "title": f"Graph 0{case_hand.curr_graph_id}",
            "xaxis_title": "Time",
            "yaxis_title": "Price",
            "font": {"size": 14, "color": "#7fafdf"},
        },
    )


@app.callback(
    Output("dummyurl", "refresh"),
    Input("submit-button", "n_clicks"),
    State("1daybounds-slider", "value"),
    State("5daybounds-slider", "value"),
    State("10daybounds-slider", "value"),
    State("30daybounds-slider", "value"),
    State("60daybounds-slider", "value"),
    State("realorrandom-dropdown", "value"),
    State("pattern-dropdown", "value"),
    State("confidence-slider", "value"),
    State("app-container", "children"),
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
    desc_children: dict,
) -> dict:
    global case_hand

    if num_clicks is None or num_clicks == 0:
        raise PreventUpdate

    case_hand.user_answers[case_hand.curr_graph_id] = {
        "1daybounds-slider": daybounds1,
        "5daybounds-slider": daybounds5,
        "10daybounds-slider": daybounds10,
        "30daybounds-slider": daybounds30,
        "60daybounds-slider": daybounds60,
        "realorrandom-dropdown": real_or_random,
        "pattern-dropdown": pattern,
        "confidence-slider": confidence,
    }

    if len(case_hand.user_answers) == TOTAL_GRAPHS:
        if len(case_hand.dataframes["1D"]) != case_hand.num_days:
            logger.debug(
                f"len(case_hand.dataframes['1D']): {len(case_hand.dataframes['1D'])} != case_hand.num_days: {case_hand.num_days}"
            )
            sys_exit(1)
        case_hand.calculate_results()
    else:
        case_hand.curr_graph_id += 1
        # ---> show results page here <---
        # ---> show results page here <---
        # ---> show results page here <---
    return desc_children


if __name__ == "__main__":
    app.run_server(debug=False, port=8080)
