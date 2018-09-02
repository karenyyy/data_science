import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff

app = dash.Dash()


figure = ff.create_table() # see docs https://plot.ly/python/table/
app.layout = html.Div([
    dcc.Graph(id='my-table', figure=figure)
])