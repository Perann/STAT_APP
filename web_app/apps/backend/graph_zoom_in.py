#import libraries
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.db import connections
from django.contrib import messages
from django.urls import reverse
import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
from ..Models.AR_Model.AR_functions import AR_model




def chart(request):
    df = pd.read_excel('./apps/tmp_data_return.xlsx', index_col=0)
    df = df.drop("Infrastructure_Equity_Listed_-_USD_Unhedged_%y/y", axis = 1)
    interpolate = df[df.columns[0]].resample('MS').interpolate(method='polynomial', order = 2)
    unsmoothed = AR_model(interpolate.values)

    #things are going a bit crazy here (that escalated quickly)
    fig = go.Figure(
        data=[
            go.Scatter( 
                x=df[df.columns[0]].dropna().index, 
                y=df[df.columns[0]].dropna(), 
                mode='lines+markers',
                line = dict(dash='0.5%'),
                name=df.columns[0],
            ),
            go.Scatter( 
                x=interpolate.dropna().index[1:], 
                y=unsmoothed[~np.isnan(unsmoothed)],
                name=df.columns[0]+" unsmoothed",
                mode='markers',
            ) 
        ],
        layout=go.Layout(
            title="Returns over time",
            width=1200,
            height=800,
            margin=dict(t=100),
            legend=dict(xanchor='right'),
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
                                'x' : [
                                    df[col].dropna().index, 
                                    df[col].dropna().resample('MS').last().index[1:] #the .last() is not important here we just want the index
                                ], 
                                #not optimized at all !!!!!!!!!! work to do (but ????? cannot dropna in a numpy array ????)
                                'y' : [
                                    df[col].dropna(), 
                                    AR_model(df[col].resample('MS').interpolate(method='polynomial', order = 2).values)[~np.isnan(AR_model(df[col].resample('MS').interpolate(method='polynomial', order = 2).values))]
                                ],
                                "name" : [
                                    col, 
                                    col+" unsmoothed"
                                ],
                            }],
                            label=col,
                            method="restyle",
                        )
                        for col in df.columns
                    ],
                    direction="down",
                    showactive=True,
                    y = 1.1,
                    yanchor = 'top',
                    pad={"r": -1000, "t": 10},
                    x = 0,
                    xanchor = 'left',
                ),
            ]
        )
    )

    fig = fig.to_html(config={'displayModeBar': False})

    context = {'fig': fig}
        
    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'home/chart.html', context=context)