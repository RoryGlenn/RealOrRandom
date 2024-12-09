from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

# Sample data for demonstration
data = pd.DataFrame(
    {
        "Date": pd.date_range(start="2023-01-01", periods=100),
        "Open": 100 + pd.Series(range(100)).apply(lambda x: x + (x * 0.05)),
        "High": 110 + pd.Series(range(100)).apply(lambda x: x + (x * 0.05)),
        "Low": 90 + pd.Series(range(100)).apply(lambda x: x + (x * 0.05)),
        "Close": 100 + pd.Series(range(100)).apply(lambda x: x + (x * 0.05)),
    }
)

# Initialize Dash app
dash_app = Dash(__name__)

# Dash layout
dash_app.layout = html.Div(
    [
        html.H1("Interactive Stock Chart with Drawing"),
        dcc.Graph(
            id="stock-chart",
            config={"editable": True},  # Enables drawing/editing
            style={"height": "600px"},
        ),
        html.Div(id="output-data"),
    ]
)


# Callback to update drawn shapes
@dash_app.callback(
    Output("output-data", "children"),
    Input("stock-chart", "relayoutData"),
)
def display_relayout_data(relayout_data):
    if relayout_data:
        return f"User added/edited: {relayout_data}"
    return "No drawing actions yet."


# Callback to render the stock chart
@dash_app.callback(Output("stock-chart", "figure"), Input("output-data", "children"))
def update_chart(_):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Stock Prices",
            )
        ]
    )
    fig.update_layout(
        title="Interactive Stock Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,  # Turn off range slider for cleaner drawing
    )
    return fig


app = dash_app
