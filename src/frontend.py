import numpy as np
from dash import dcc, html
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# https://dash-example-index.herokuapp.com/checklist-plots


class FrontEnd:
    """Contains div elements and logic needed to create the front end"""

    dataframes = {}
    half_dataframes = {}
    timeframe_map = {
        # "1m": "1min",
        # "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "1D": "1D",
        "3D": "3D",
        "W": "1W",
        "M": "1M",
    }

    bounds = np.round(np.linspace(-100, 100, 2001), 1)

    @staticmethod
    def get_graph_layout(fig: go.Figure) -> html.Div:
        """Updates the layout for the graph figure"""
        return html.Div(
            [
                dcc.Graph(
                    id="graph-main",
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
                    style={"width": "155vh", "height": "90vh"},
                )
            ]
        )

    @staticmethod
    def get_app_layout() -> html.Div:
        return html.Div(
            [
                # dcc.Link(html.Button("LOG_VIEW"), href="/log_stream", refresh=True, id='loadnewpage'),
                # The memory store reverts to the default on every page refresh
                # dcc.Store(id="submit", storage_type='session', data=dict),
                dcc.Store(id="submit"),
                html.H1("Chart"),
                dbc.Row(
                    [
                        # timeframe dropdown
                        dbc.Col(FrontEnd.get_timeframe_dropdown()),
                        # loading bar
                        # dbc.Spinner(
                        #     children=[dcc.Graph(id="graph-main")],
                        #     fullscreen=True,
                        # ),
                    ]
                ),
                dbc.Row(),
                html.Hr(),
                html.Div(id="page-content"),
                # Bounds dropdown
                html.Div(
                    dbc.Row(
                        [
                            # 1 day
                            FrontEnd.get_bounds_slider(
                                text="What will the price be 1 day after the last candle bars close price?",
                                id="1daybounds-slider",
                            ),
                        ]
                    ),
                ),
                html.Div(
                    dbc.Row(
                        [
                            # 5 days
                            FrontEnd.get_bounds_slider(
                                text="What will the price be 5 days after the last candle bars close price?",
                                id="5daybounds-slider",
                            ),
                        ]
                    ),
                ),
                html.Div(
                    dbc.Row(
                        [
                            # 10 days
                            FrontEnd.get_bounds_slider(
                                text="What will the price be 10 days after the last candle bars close price?",
                                id="10daybounds-slider",
                            ),
                        ]
                    ),
                ),
                html.Div(
                    dbc.Row(
                        [
                            # 30 days
                            FrontEnd.get_bounds_slider(
                                text="What will the price be 30 days after the last candle bars close price?",
                                id="30daybounds-slider",
                            ),
                        ]
                    ),
                ),
                html.Div(
                    dbc.Row(
                        [
                            # 60 days
                            FrontEnd.get_bounds_slider(
                                text="What will the price be 60 days after the last candle bars close price?",
                                id="60daybounds-slider",
                            ),
                        ],
                    ),
                ),
                # Real or Random
                FrontEnd.get_real_or_random_dropdown(),
                # pattern
                FrontEnd.get_pattern_dropdown(),
                # confidence
                FrontEnd.get_confidence_slider(),
                FrontEnd.submit_button(),
            ],
            style={
                "margin-left": "5%",
                "margin-right": "5%",
                "margin-top": "20px",
                "margin-bottom": "200px",
            },
        )

    @staticmethod
    def get_timeframe_dropdown() -> html.Div:
        return html.Div(
            [
                html.P("Timeframe"),
                dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[
                        {"label": timeframe, "value": timeframe}
                        for timeframe in FrontEnd.timeframe_map
                    ],
                    value="1D",
                ),
            ],
            style={"width": "10%", "margin-bottom": "10px"},
        )

    @staticmethod
    def get_bounds_slider(text: str, id: str) -> html.Div:
        """What are the upper bounds and what are the lower bounds for your price
        prediction 1, 5, 10, 30, 60 bars from the last candle bar?"""
        return html.Div(
            [html.P(text), FrontEnd.bounds_slider(id)],
            style={"width": "50%", "margin-bottom": "20px"},
        )

    @staticmethod
    def get_real_or_random_dropdown() -> html.Div:
        """Is this graph real or random?"""
        return html.Div(
            [
                html.P("Is this graph real or random?"),
                dcc.Dropdown(
                    id="realorrandom-dropdown",
                    options=["Real", "Random"],
                    value="",
                ),
            ],
            style={"width": "25%", "margin-bottom": "30px"},
        )

    @staticmethod
    def get_pattern_dropdown() -> html.Div:
        """What pattern do you see?"""
        return html.Div(
            [
                html.P("What pattern do you see?"),
                dcc.Dropdown(
                    options=FrontEnd.__get_candle_stick_patterns(),
                    id="pattern-dropdown",
                ),
            ],
            style={"width": "25%", "margin-bottom": "30px"},
        )

    @staticmethod
    def get_confidence_slider() -> html.Div:
        """How confident are you overall in your answers?"""
        return html.Div(
            [
                html.P("What is your overall confidence in your answers?"),
                html.P("(0: not at all confident, 10: extremely confident)"),
                # html.P("10: extremely confident"),
                dcc.Slider(
                    id="confidence-slider",
                    min=0,
                    max=10,
                    step=1,
                    value=5,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={"width": "25%", "margin-bottom": "30px"},
        )

    @staticmethod
    def submit_button() -> html.Div:
        return html.Div(
            [
                dcc.ConfirmDialogProvider(
                    html.Th(
                        html.Button(
                            id="submit-button", children="Submit", type="button"
                        )
                    ),
                    id="submit-provider",
                    message="You will not be able to go back after submitting.\nAre you sure you want to continue?",
                ),
                html.Div(id="submit_output-provider"),
            ],
            style={"width": "25%", "margin-bottom": "30px"},
        )
        # return (
        #     dcc.Link(
        #         html.Button("Submit"),
        #         href="/log_stream",
        #         refresh=True,
        #         id="loadnewpage",
        #     ),
        # )

    def bounds_slider(id: str) -> html.Div:
        return html.Div(
            [
                dcc.Slider(
                    id=id,  # any name you'd like to give it
                    step=0.5,  # number of steps between values
                    marks={
                        -100: {"label": "-100%", "style": {"color": "#f50"}},
                        0: {"label": "0%", "style": {"color": "#0f1000"}},
                        100: {"label": "100%", "style": {"color": "#77b0b1"}},
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
            style={"width": "50%", "margin-bottom": "30px"},
        )

    def __get_candle_stick_patterns() -> list[str]:
        return [
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
        ]
