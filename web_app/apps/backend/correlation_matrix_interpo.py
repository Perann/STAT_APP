
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
from ..Models.AR_Model.AR_functions import AR_model


def correlation_matrix_interpo(request):
    # Importing the dataset
    alternative_asset_data = pd.read_excel('./apps/tmp_data_return.xlsx')
    alternative_asset_data = alternative_asset_data.drop(['Infrastructure_Equity_Listed_-_USD_Unhedged_%y/y'], axis = 1)
    alternative_asset_data.rename(lambda c: c.replace('returns_',''), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('_%y/y',''), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('_',' '), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('USD Unhedged',''), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('USD Hedged',''), axis = 1, inplace = True)

    alternative_asset_data.rename(lambda c: c.replace('-',''), axis = 1, inplace = True)
    alternative_asset_data.rename(lambda c: c.replace('  ',' '), axis = 1, inplace = True)

    # Preprocessing
    alternative_asset_data = alternative_asset_data.set_index('QUARTER')
    n = len(alternative_asset_data.columns)
    def data_unsmoothing_linear(df_):
        new = df_.resample('MS').interpolate(method='linear')
        res = pd.DataFrame(index = new.index)
        for col in df_.columns:
            res[col] = AR_model(new[col].values)
        return res


    unsmoothed_linear = data_unsmoothing_linear(alternative_asset_data)
    unsmoothed_linear = unsmoothed_linear.dropna()
    corr = unsmoothed_linear.corr()
    styles = [dict(selector="th", props=[("font-size", "12px")])]
    corr = corr.style.format(precision = 2).background_gradient(cmap = 'RdYlGn', axis = None).set_table_styles(styles)
    safety = {'HTML' : corr.to_html(bold_rows = False)}
    return JsonResponse(safety)