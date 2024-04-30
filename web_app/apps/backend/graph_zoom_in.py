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
from ..Models.getmansky.GetmanskyMain import GetmanskyModel


def tweak_alternative_data(df_):
    return (df_.assign(**{col + '_%y/y': df_[col].pct_change(fill_method = None) for col in df_.columns})
            .rename(lambda c: c.replace(' ','_'), axis = 1)) #Adding the returs

def chart(request):
    df = pd.read_excel('./apps/tmp_data_return.xlsx', index_col=0)
    df = df.drop("Infrastructure_Equity_Listed_-_USD_Unhedged_%y/y", axis = 1)
    interpolate = df[df.columns[0]].resample('MS').interpolate(method='polynomial', order = 2)
    unsmoothed = AR_model(interpolate.values)

    # classic_data = pd.read_excel('./apps/tmp_classic_data_return.xlsx', index_col=0)
    # results = classic_data.copy()
    # results = results.merge(df[["Private_Equity_USD_Unhedged_%y/y"]], how = 'inner', left_index = True, right_index = True)
    # results = results[1:]
    # bench = results["US_Equity_USD_Unhedged_%y/y"]
    # getmansky = GetmanskyModel(2)
    # getmansky.fit(bench.values.reshape(-1, 1), results["Private_Equity_USD_Unhedged_%y/y"].values.reshape(-1,1))
    # unsmoothed_getmansky = getmansky.predict(bench)
    # #print(unsmoothed_getmansky)
    classic_asset_data = (
            # Importing the dataset
            pd.read_excel("/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx", sheet_name= "Classic Asset")
            # Preprocessing
            .filter(['QUARTER', 'Date', 'US Equity USD Unhedged'])
            .dropna()
            .set_index('Date', drop = False)
            .resample('M')
            .last()
            .dropna()
            .assign(returns_US_equity = (lambda x: x['US Equity USD Unhedged'].pct_change(fill_method=None)))
            .dropna()
            .set_index("QUARTER")
        )

    def _helper(bench, alter):
        alternative_asset_data = (
            # Importing the dataset
            pd.read_excel("/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx", sheet_name= "Alternative Asset", index_col=0))

        alternative_data = tweak_alternative_data(alternative_asset_data)
        alternative_data = alternative_data[[col for col in alternative_data.columns if '%y/y' in col]]

        classic_asset_data = (
            # Importing the dataset
            pd.read_excel("/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx", sheet_name= "Classic Asset")
            # Preprocessing
            #.filter(['QUARTER', 'Date', 'US Equity USD Unhedged'])
            .dropna()
            .set_index('Date', drop = False)
            .resample('M')
            .last()
            .dropna()
            .assign(returns_US_equity = (lambda x: x['US Equity USD Unhedged'].pct_change(fill_method=None)))
            .dropna()
            .set_index("QUARTER")
        )

        results = classic_asset_data.copy()
        results = results.merge(alternative_data, how = 'inner', left_index = True, right_index = True)
        results = results[1:]

        getmansky = GetmanskyModel(2)
        getmansky.set_default_weights("sumOfYears")
        getmansky.fit(results[bench].values.reshape(-1, 1), results[alter].values.reshape(-1,1))
        results['returns unsmoothed'] = getmansky.predict(results[bench])

        results = results.set_index('Date')
        return results.index, results['returns unsmoothed']
    
    bench = 'returns_US_equity'
    alter = 'Global_Property_USD_Unhedged_%y/y'
    #for line in results.iterrows():
    #    if line[0].month in [1, 2, 4, 5, 7, 8, 10, 11]:
    #        line[1]['returns_PE'] = None
    


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
            ),
            go.Scatter( 
                x=_helper(bench, alter)[0], 
                y=_helper(bench, alter)[1],
                name="test",
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
                title='Date',
                showgrid=True,
                gridcolor='#191919',
                rangeslider=dict(
                    visible=True,
                ),    
            ),
            yaxis=dict(
                title='Returns',
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
                                    df[col].dropna().resample('MS').last().index[1:], #the .last() is not important here we just want the index
                                    _helper(bench, col)[0]
                                ], 
                                #not optimized at all !!!!!!!!!! work to do (but ????? cannot dropna in a numpy array ????)
                                'y' : [
                                    df[col].dropna(), 
                                    AR_model(df[col].resample('MS').interpolate(method='polynomial', order = 2).values)[~np.isnan(AR_model(df[col].resample('MS').interpolate(method='polynomial', order = 2).values))],
                                    _helper(bench, col)[1]
                                    ],
                                "name" : [
                                    col, 
                                    col + " unsmoothed",
                                    col + " unsmoothed getmansky"
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
                dict(
                    buttons=[
                        dict(
                            args=[{
                                
                            }],
                            label=col,
                            method="restyle",
                        )
                        for col in classic_asset_data.columns
                    ],
                    direction="down",
                    showactive=True,
                    y = 1.1,
                    yanchor = 'top',
                    pad={"r": -1000, "t": 10},
                    x = 0.5,
                    xanchor = 'left',
                ),
            ]
        )
    )

    fig = fig.to_html(config={'displayModeBar': False})

    context = {'fig': fig}
        
    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'home/chart.html', context=context)