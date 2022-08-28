import numpy as np
from dash import dcc, html
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


# https://dash.plotly.com/dash-core-components/store


class FrontEnd:
    """Contains div elements and logic needed to create the front end"""

    dataframes = {}
    half_dataframes = {}
    timeframe_map = {
        # "1_Minute": "1min",
        # "5_Minute": "5min",
        "15_Minute": "15min",
        "30_Minute": "30min",
        "1_Hour": "1H",
        "2_Hour": "2H",
        "4_Hour": "4H",
        "1_Day": "1D",
        "1_Week": "1W",
        "1_Month": "1M",
    }

    upper_bounds = np.round(np.linspace(0.1, 100, 1000), 1)
    lower_bounds = np.round(np.linspace(-0.1, -100, 1000), 1)

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
                html.H1("Chart"),
                dbc.Row(
                    [
                        # timeframe dropdown
                        dbc.Col(FrontEnd.get_timeframe_dropdown())
                    ]
                ),
                dbc.Row(),
                html.Hr(),
                html.Div(id="page-content"),
                # Bounds dropdown
                # 1 day
                FrontEnd.get_bounds_dropdown(
                    "What will the upper bounds be 1 day after the last candle bar?",
                    "1dayupperbounds-dropdown",
                    True,
                ),
                FrontEnd.get_bounds_dropdown(
                    "What will the lower bounds be 1 day after the last candle bar?",
                    "1daylowerbounds-dropdown",
                    False,
                ),
                # 5 days
                FrontEnd.get_bounds_dropdown(
                    "What will the upper bounds be 5 days after the last candle bar?",
                    "5dayupperbounds-dropdown",
                    True,
                ),
                FrontEnd.get_bounds_dropdown(
                    "What will the lower bounds be 5 days after the last candle bar?",
                    "5daylowerbounds-dropdown",
                    False,
                ),
                # 10 days
                FrontEnd.get_bounds_dropdown(
                    "What will the upper bounds be 10 days after the last candle bar?",
                    "10dayupperbounds-dropdown",
                    True,
                ),
                FrontEnd.get_bounds_dropdown(
                    "What will the lower bounds be 10 days after the last candle bar?",
                    "10daylowerbounds-dropdown",
                    False,
                ),
                # 30 days
                FrontEnd.get_bounds_dropdown(
                    "What will the upper bounds be 30 days after the last candle bar?",
                    "30dayupperbounds-dropdown",
                    True,
                ),
                FrontEnd.get_bounds_dropdown(
                    "What will the lower bounds be 30 days after the last candle bar?",
                    "30daylowerbounds-dropdown",
                    False,
                ),
                # 60 days
                FrontEnd.get_bounds_dropdown(
                    "What will the upper bounds be 60 days after the last candle bar?",
                    "60dayupperbounds-dropdown",
                    True,
                ),
                FrontEnd.get_bounds_dropdown(
                    "What will the lower bounds be 60 days after the last candle bar?",
                    "60daylowerbounds-dropdown",
                    False,
                ),
                # Real or Random
                FrontEnd.get_real_or_random_dropdown(),
                # pattern
                FrontEnd.get_pattern_textbox(),
                # confidence
                FrontEnd.get_confidence_slider(),
            ],
            style={"margin-left": "5%", "margin-right": "5%", "margin-top": "20px"},
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
                    value="1_Day",
                ),
            ]
        )

    @staticmethod
    def get_bounds_dropdown(text: str, id: str, upper: bool) -> html.Div:
        """What are the upper bounds and what are the lower bounds for your price
        prediction 1, 5, 10, 30, 60 bars from the last candle bar?"""
        return html.Div(
            [
                html.P(text),
                dcc.Dropdown(
                    id=id,
                    options=[{"label": i, "value": i} for i in FrontEnd.upper_bounds]
                    if upper
                    else [{"label": i, "value": i} for i in FrontEnd.lower_bounds],
                    # value=0.1 if upper else -0.1,
                    value="",
                ),
            ]
        )

    @staticmethod
    def get_real_or_random_dropdown() -> html.Div:
        # Is this graph real or random?
        return html.Div(
            [
                html.P("Is this graph real or random?"),
                dcc.Dropdown(
                    id="realorrandom-dropdown", options=["Real", "Random"], value=""
                ),
            ]
        )

    @staticmethod
    def get_pattern_textbox() -> html.Div:
        # Do you see a recognizable pattern in the graph? If so, what pattern is it?
        return html.Div(
            [
                html.P(
                    "Do you see a recognizable pattern in the graph? If so, what pattern is it?"
                ),
                dcc.Textarea(
                    id="pattern-textbox",
                ),
            ]
        )

    @staticmethod
    def get_confidence_slider() -> html.Div:
        # How confident are you overall in your answers?
        return html.Div(
            [
                html.P("Rate your overall confidence in your answers"),
                dcc.RangeSlider(id="confidence-slider", min=0, max=100, step=10),
            ]
        )
