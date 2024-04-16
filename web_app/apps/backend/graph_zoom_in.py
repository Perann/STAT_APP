#import libraries
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.db import connections
from django.contrib import messages
from django.urls import reverse
import pandas as pd
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
import os
from ....AR_Model.AR_functions import AR_model



def chart(request):
    df = pd.read_excel('./tmp_data_return.xlsx', index_col=0)
    df = df.drop("Infrastructure_Equity_Listed_-_USD_Unhedged_%y/y", axis = 1)
    interpolate = df[df.columns[0]].resample('MS').interpolate(method='polynomial', order = 2)
    unsmoothed = AR_model(interpolate.values)

    
    fig = go.Figure(
        data=[
            go.Scatter( 
                x=df.index, 
                y=df[df.columns[0]], 
                mode='lines',
                name = df.columns[0],
            ),
            go.Scatter( 
                x=df.index, 
                y=unsmoothed, 
                mode="lines",
                name = df.columns[0]+" unsmoothed",
            ) 
        ],
        layout=go.Layout(
            title='Caca',
            xaxis=dict(
                title='X Axis',
                showgrid=True,
                gridcolor='#191919',
                rangeslider=dict(
                    visible=True,
                ),    
            ),
            yaxis=dict(
                title='Y Axis',
                showgrid=True,
                gridcolor='#191919',
            ),
            plot_bgcolor='#ffffff',
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{
                                "y": [df[col], AR_model(df[col].resample('MS').interpolate(method='polynomial', order = 2).values)],
                                "name" : [col, col+" unsmoothed"],
                            }],
                            label=col,
                            method="restyle",
                        )
                        for col in df.columns
                    ],
                    direction="down",
                ),
            ]
        )
    )

    fig = fig.to_html()

    context = {'fig': fig}
        
    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'home/chart.html', context=context)