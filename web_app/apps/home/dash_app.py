# dash_app.py
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
import warnings
import pandas as pd
from dash import dash_table


app = DjangoDash('SimpleExample')  # Remplace dash.Dash

df = pd.read_excel('./tmp_data_return.xlsx').round(2)
filtered_columns = [col for col in df.columns if df[col].dtypes ==  "datetime64[ns]"]
df[filtered_columns] = df[filtered_columns].apply(lambda x: x.dt.strftime('%Y-%m-%d'))

app.layout = dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[
        {'id': c, 'name': c} 
        for c in df.columns
    ],
    style_header={'backgroundColor': 'rgb(60, 65, 105)', "color": "white"},
)