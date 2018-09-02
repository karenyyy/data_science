import plotly
import random
import plotly.graph_objs as go

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Event

from collections import deque

X = deque(maxlen=20)
y = deque(maxlen=20)

X.append(1)
y.append(1)

app = dash.Dash('live-update')

app.layout = html.Div(
    [
        dcc.Graph(id='live-update',
                  animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1000
        ),
    ]
)


@app.callback(output=Output(component_id='live-update',
                            component_property='figure'),
              events=[Event(component_id='graph-update',
                            component_event='interval')])
def update_graph():
    global X
    global y
    X.append(X[-1] + 1)
    y.append(y[-1] + y[-1] * random.uniform(-0.1, 0.1))

    data = go.Scatter(
        x=list(X),
        y=list(y),
        name='Scatter',
        mode='lines+markers'
    )

    return {'data': [data],
            'layout': go.Layout(
                xaxis=dict(range=[min(X), max(X)]),
                yaxis=dict(range=[min(y), max(y)]))}


if __name__ == '__main__':
    app.run_server(debug=True)
