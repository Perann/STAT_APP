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
    alternative_asset_data = pd.read_excel('/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')

    # Preprocessing
    alternative_asset_data = alternative_asset_data[[a for a in list(alternative_asset_data.keys()) if a != 'Infrastructure Equity Listed - USD Unhedged']]
    alternative_asset_data.dropna(inplace = True)
    list_key_return = []
    for key in alternative_asset_data.keys():
        if key != 'QUARTER':
            alternative_asset_data[f'returns {key}'] = alternative_asset_data[key].pct_change(fill_method=None)
            list_key_return.append(f'returns {key}')
    alternative_asset_data.dropna(inplace = True)
    print(alternative_asset_data)
    alternative_asset_data = alternative_asset_data.set_index('QUARTER')
    alternative_asset_data = alternative_asset_data[list_key_return]
    alternative_asset_data.rename(lambda c: c.replace('returns',''), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('USD Unhedged',''), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('-',''), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('USD Hedged',''), axis = 1, inplace = True)
    corr = alternative_asset_data.corr()
    styles = [dict(selector="th", props=[("font-size", "12px")])]
    corr = corr.style.format(precision = 2).background_gradient(cmap = 'RdYlGn', axis = None).set_table_styles(styles)
    safety = {'HTML' : corr.to_html(bold_rows = False)}
    return JsonResponse(safety)