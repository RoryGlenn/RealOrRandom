# # -*- coding: utf-8 -*-
# from dash import Dash, dcc, html
# import time

# from dash.dependencies import Input, Output

# app = Dash(__name__)

# app.layout = html.Div(
#     children=[
#         html.H3("Edit text input to see loading state"),
#         dcc.Input(id="loading-input-1", value="Input triggers local spinner"),
#         dcc.Loading(
#             id="loading-1", type="default", children=html.Div(id="loading-output-1")
#         ),
#     ],
# )


# @app.callback(Output("loading-output-1", "children"), Input("loading-input-1", "value"))
# def input_triggers_spinner(value):
#     time.sleep(10)
#     return value


# if __name__ == "__main__":
#     app.run_server(debug=True)
