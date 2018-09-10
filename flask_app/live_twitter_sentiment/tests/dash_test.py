import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

X = deque(maxlen=20)
X.append(1)
y = deque(maxlen=20)
y.append(1)

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc
    ]
)