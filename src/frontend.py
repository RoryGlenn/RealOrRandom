import pandas as pd
from dash import dcc, html
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


class FrontEnd:
    timeframe_map: dict[str, str] = {
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

    @staticmethod
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

    @staticmethod
    def get_timeframe_dropdown(timeframes: list[str]) -> html.Div:
        return html.Div(
            [
                html.P("Timeframe"),
                dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[
                        {"label": timeframe, "value": timeframe}
                        for timeframe in timeframes
                    ],
                    value="1_Day",
                ),
            ]
        )

    @staticmethod
    def get_config() -> dict:
        """Returns the basic config options at the top right of the graph"""
        return {
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
        }

    @staticmethod
    def get_graph_layout(fig: go.Figure) -> html.Div:
        """Updates the layout for the graph figure"""

        return html.Div(
            [
                dcc.Graph(
                    figure=fig,
                    config=FrontEnd.get_config(),
                )
            ]
        )


    @staticmethod
    def app_update_layout():
        return

    @staticmethod
    def app_create_layout(
    ) -> html.Div:
        """Creates the layout for the entire page"""

        return html.Div(
            [
                html.H1("Real Time Charts"),
                dbc.Row(
                    [
                        dbc.Col(
                            FrontEnd.get_timeframe_dropdown(list(FrontEnd.timeframe_map.keys()))
                        ),
                    ]
                ),
                dbc.Row(),
                html.Hr(),
                # This needs to be changed to the correct input function!!!
                dcc.Interval(id="update", interval=5000),
                # the entire page content to be loaded, callback function needed for this!
                html.Div(id="page-content"),
            ],
            style={"margin-left": "5%", "margin-right": "5%", "margin-top": "20px"},
        )
