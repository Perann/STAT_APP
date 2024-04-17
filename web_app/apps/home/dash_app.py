# dash_app.py
from django_plotly_dash import DjangoDash
import warnings
import pandas as pd
import dash
from dash import Dash, dash_table, dcc, html, Input, Output, State, callback
import base64
import io
from django.shortcuts import render


app = DjangoDash('SimpleExample')
#app = Dash('SimpleExample')

df = pd.read_excel('./apps/tmp_data_return.xlsx').round(2)
#df = pd.read_excel("/Users/adamelbernoussi/Desktop/GitHub_repos/STAT_APP/web_app/tmp_data_return.xlsx").round(2)
filtered_columns = [col for col in df.columns if df[col].dtypes ==  "datetime64[ns]"]
df[filtered_columns] = df[filtered_columns].apply(lambda x: x.dt.strftime('%Y-%m-%d'))

app.layout = html.Div([
    dcc.Upload(
        id='datatable-upload',
        children=html.Div([
            'Drag and Drop or ',
            html.B('Select Files')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
    ),
    dash_table.DataTable(
        id = 'datatable-upload-table',
        data=df.to_dict('records'),
        columns=[
            {'id': c, 'name': c} 
            for c in df.columns
        ],
        style_header={'backgroundColor': 'rgb(60, 65, 105)', "color": "white"},
    )
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(io.BytesIO(decoded))
    elif 'xlsx' in filename:
        return pd.read_excel(io.BytesIO(decoded))


@callback([dash.dependencies.Output('datatable-upload-table', 'data'),
              dash.dependencies.Output('datatable-upload-table', 'columns')],
              [dash.dependencies.Input('datatable-upload', 'contents')],
              #[dash.dependencies.State('datatable-upload', 'filename')],
              prevent_initial_call=True)
def update_output(contents, filename, **kwargs):
    if contents is None:
        return [{}], []
    df = parse_contents(contents, filename)
    #df.to_excel('./tmp.xlsx')
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]

if __name__ == '__main__':
    app.run_server(debug=True)