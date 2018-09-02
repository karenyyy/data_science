import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import re
import plotly.figure_factory as ff


desc_file = 'airq402descrip.txt'
input_file = 'airq402dat.csv'

class CustomAirlineVis(object):
    def __init__(self):
        self.desc_file = desc_file
        self.input_file = input_file
        self.col_names = []

        self.get_col_name()

        self.data = pd.read_csv(filepath_or_buffer=self.input_file,
                                sep=',',
                                header=None,
                                names=self.col_names)

        self.app = dash.Dash()
        self.dash_vis()

    def detect_empty_line(self, array):
        return np.where(list(map(lambda x: len(x) > 1, array)))[0]

    def detect_number(self, array):
        return np.where(list(map(lambda x: re.match("^[a-zA-Z(]+.*", x), array)))[0]

    def get_col_name(self, start_line=5):
        with open(file=self.desc_file,
                  encoding='utf-8',
                  mode='r') as f:
            count = 0
            for line in f:
                if line == '\n':
                    continue
                if count >= start_line:
                    line = np.array(line.split(' '))
                    line = line[self.detect_empty_line(line)]
                    line = line[self.detect_number(line)]
                    col_name = ' '.join(line)
                    self.col_names.append(col_name)
                count += 1

    def dash_vis(self):

        figure = ff.create_table(self.data[:10])
        self.app.layout = html.Div([
            html.H1('Dash Visualization Tool:\n Test 1'),
            dcc.Graph(id='my-table', figure=figure),
            dcc.Graph(
                id='1',
                figure={
                    'data': [
                        {'x': self.data[self.col_names[0]],
                         'y': self.data[self.col_names[2]],
                         'type': 'bar',
                         'name': 'self.data1'}
                    ],
                    'layout': {
                        'title': '{} - {}'.format(self.col_names[0], self.col_names[2])
                    }
                }
            ),
            dcc.Graph(
                id='2',
                figure={
                    'data': [
                        {'x': self.data[self.col_names[0]],
                         'y': self.data[self.col_names[3]],
                         'type': 'bar',
                         'name': 'self.data2'}
                    ],
                    'layout': {
                        'title': '{} - {}'.format(self.col_names[0], self.col_names[3])
                    }
                }
            ),
            dcc.Graph(
                id='3',
                figure={
                    'data': [
                        {'x': self.data[self.col_names[0]],
                         'y': self.data[self.col_names[4]],
                         'type': 'bar',
                         'name': 'self.data3'}
                    ],
                    'layout': {
                        'title': '{} - {}'.format(self.col_names[0], self.col_names[4])
                    }
                }
            )
        ])


if __name__ == '__main__':
    custom = CustomAirlineVis()
    custom.app.run_server(debug=True)
