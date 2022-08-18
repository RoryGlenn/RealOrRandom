from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import plotly.graph_objects as go
import pandas as pd


class FrontEnd:
    timeframes: list[str] = [
    "1 Minute",
    "5 Minute",
    "15 Minute",
    "30 Minute",
    "1 Hour",
    "2 Hour",
    "4 Hour",
    "1 Day",
    "1 Week",
]
    # def __init__(self) -> None:
        # self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


    @staticmethod
    def get_config() -> dict:
        return (
            {
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
        )

    @staticmethod
    def create_figure(df: pd.DataFrame, graph_title: str) -> go.Figure:
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

    # @app.callback(
    #     Output("page-content", "children"),
    #     Input("update", "n_intervals"),
    #     State("symbol-dropdown", "value"),
    #     State("timeframe-dropdown", "value"),
    #     State("num-bar-input", "value"),
    # )
    # def update_ohlc_chart(self, timeframe: str, timeframes: list[str], symbol: str, num_bars: str):
    #     timeframe_str = timeframe
    #     timeframe = timeframes
    #     num_bars = int(num_bars)

    #     print(symbol, timeframe, num_bars)

    #     bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    #     df = pd.DataFrame(bars)
    #     df["time"] = pd.to_datetime(df["time"], unit="s")

    #     fig = go.Figure(
    #         data=go.Candlestick(
    #             x=df["time"],
    #             open=df["open"],
    #             high=df["high"],
    #             low=df["low"],
    #             close=df["close"],
    #         )
    #     )

    #     fig.update(layout_xaxis_rangeslider_visible=False)
    #     fig.update_layout(yaxis={"side": "right"})
    #     fig.layout.xaxis.fixedrange = True
    #     fig.layout.yaxis.fixedrange = True

    #     return [
    #         html.H2(id="chart-details", children=f"{symbol} - {timeframe_str}"),
    #         dcc.Graph(figure=fig, config={"displayModeBar": False}),
    #     ]

    @staticmethod
    def get_timeframe_dropdown(timeframes: list[str]) -> html.Div:
        return html.Div(
            [
                html.P("Timeframe:"),
                dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[
                        {"label": timeframe, "value": timeframe}
                        for timeframe in timeframes
                    ],
                    value="D1",
                ),
            ]
        )

    @staticmethod
    def get_num_bars_input() -> html.Div:
        return html.Div(
            [
                html.P("Number of Candles"),
                dbc.Input(id="num-bar-input", type="number", value="20"),
            ]
        )

    @staticmethod
    def app_update_layout(fig: go.Figure) -> html.Div:
        """Updates the layout for the graph figure"""
        return html.Div(
            [
                dcc.Graph(
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
                )
            ]
        )

    @staticmethod
    def app_create_layout(
        fig: go.Figure,
        # timeframes: list[str] = [
        #     "1 Minute",
        #     "5 Minute",
        #     "15 Minute",
        #     "30 Minute",
        #     "1 Hour",
        #     "2 Hour",
        #     "4 Hour",
        #     "1 Day",
        #     "1 Week",
        # ],
    ) -> html.Div:

        # creates the layout of the App
        layout = html.Div(
            [
                html.H1("Real Time Charts"),
                dbc.Row(
                    [
                        # dbc.Col(symbol_dropdown),
                        dbc.Col(FrontEnd.get_timeframe_dropdown(FrontEnd.timeframes)),
                        dbc.Col(FrontEnd.get_num_bars_input()),
                    ]
                ),
                dbc.Row(FrontEnd.app_update_layout(fig)),
                html.Hr(),
                # dcc.Interval(id="update", interval=200),
                html.Div(id="page-content"),
            ],
            style={"margin-left": "5%", "margin-right": "5%", "margin-top": "20px"},
        )
        return layout
