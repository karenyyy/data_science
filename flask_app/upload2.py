import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Upload(html.Button('Upload File')),

    html.Hr(),

    dcc.Upload(html.A('Upload File')),

    html.Hr(),

    dcc.Upload([
        'Drag and Drop or ',
        html.A('Select a File')
    ], style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
    })
])

if __name__ == '__main__':
    app.run_server(debug=True)