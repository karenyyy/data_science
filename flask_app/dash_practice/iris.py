import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html


from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

x_idx = list(range(len(X)))
legends = ['iris type 1', 'iris type 2',
           'iris type 3', 'iris type 4']

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1('Dash Iris'),
    dcc.Input(id='input',
              value='Enter something:',
              type='text'),
    html.Div(id='output'),
    dcc.Graph(
        id='iris',
        figure={
            'data': [
                {'x': x_idx,
                 'y': X[:, 0],
                 'type': 'line',
                 'name': legends[0]},
                {'x': x_idx,
                 'y': X[:, 1],
                 'type': 'line',
                 'name': legends[1]},
                {'x': x_idx,
                 'y': X[:, 2],
                 'type': 'line',
                 'name': legends[2]},
                {'x': x_idx,
                 'y': X[:, 3],
                 'type': 'line',
                 'name': legends[3]},
            ],
            'layout': {
                'title': 'Iris Dash Example'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
