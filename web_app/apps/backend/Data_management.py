"""
In this file, we implement a function to load the data and display it in a table.
"""

# Importing libraries
import pandas as pd
import plotly.figure_factory as ff
from django.shortcuts import render


def load_data(request):
    df = pd.read_excel('./tmp_data_return.xlsx', index_col=0).round(2)
    table = ff.create_table(df)

    table = table.to_html()

    context = {'table': table}
        
    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'home/tables.html', context=context)