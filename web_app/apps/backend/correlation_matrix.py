"""In this file we will implement a function to compute the correlation matrix of the returns of the assets."""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from django.http import JsonResponse

def correlation_matrix(request):
    # Importing the dataset
    corr = (
        pd.read_excel('/Users/adamelbernoussi/Desktop/GitHub_repos/STAT_APP/web_app/apps/tmp_data_return.xlsx')
        .drop(['Infrastructure_Equity_Listed_-_USD_Unhedged_%y/y'], axis = 1)
        .rename(lambda c: c.replace('returns_',''), axis = 1)
        .rename(lambda c: c.replace('_%y/y',''), axis = 1)
        .rename(lambda c: c.replace('_',' '), axis = 1)
        .rename(lambda c: c.replace('USD Unhedged',''), axis = 1)
        .rename(lambda c: c.replace('USD Hedged',''), axis = 1)
        .rename(lambda c: c.replace('-',''), axis = 1)
        .rename(lambda c: c.replace('  ',' '), axis = 1)
        .set_index("QUARTER")
        .corr()
        .style
            .format(precision = 2)
            .background_gradient(cmap = 'RdYlGn', axis = None)
            .set_table_styles(
                [dict(selector="th", props=[("font-size", "12px")])]
            )
    )

    safety = {'HTML' : corr.to_html(bold_rows = False)}
    return JsonResponse(safety)