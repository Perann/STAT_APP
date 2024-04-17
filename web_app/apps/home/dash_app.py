# dash_app.py
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html

app = DjangoDash('SimpleExample')  # Remplace dash.Dash

app.layout = html.Div([
    html.H1('Bonjour Dash dans Django!'),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [{'x': [1, 2, 3], 'y': [4, 3, 1], 'type': 'bar', 'name': 'SF'}],
            'layout': {'title': 'Dash Data Visualization'}
        }
    )
], style={'padding-bottom': "100hv"})
