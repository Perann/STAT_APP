import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from AR_functions import get_alpha, get_returns, get_gamma_phi, AR_model

#Plotting

def ResultViz(time, smoothed_data, unsmoothed_data, asset = 'Default'):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    fig.suptitle('Yield Analysis, return ' + asset + ' unsmoothed by AR', fontsize=16)

    ax1.plot(time,unsmoothed_data, label = 'Unsmoothed Returns')
    ax1.plot(time,smoothed_data, label = 'Observed Returns')
    ax1.legend()
    ax1.set_xticks(time[::15])
    ax1.set_xlabel('Quarters')
    ax1.set_ylabel('Returns')

    cum_smoothed= np.cumsum(smoothed_data)
    cum_unsmoothed = np.cumsum(unsmoothed_data)

    ax2.plot(time, cum_unsmoothed,label  = 'Unsmoothed cumulated returns')
    ax2.plot(time,cum_smoothed,label = 'Observed cumulated returns')
    ax2.legend()
    ax2.set_xticks(time[::15])
    ax2.set_xlabel('Quarters')
    ax2.set_ylabel('Cumulated returns')

    plt.show()

def data_prep(dataframe, key):
    dataframe = dataframe[['QUARTER', key]].dropna()
    quarter = dataframe['QUARTER']
    smooth = dataframe[key].reset_index(drop = True)
    unsmoothed = AR_model(smooth)
    return quarter, smooth, unsmoothed

def get_autocorrel(smoothed_data, unsmoothed_data, lag = 1):
    pd_smooth = pd.Series(smoothed_data)
    pd_unsmooth = pd.Series(unsmoothed_data)
    res = (pd_smooth.autocorr(k), pd_unsmooth.autocorr(lag))
    return res

if __name__ == '__main__':
    
    #Preprocessing
    alternative_asset_data = pd.read_excel('EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
    for key in alternative_asset_data.keys()[1:]:
        alternative_asset_data['Return ' + key] = alternative_asset_data[key].pct_change()
    
    
    print(alternative_asset_data.keys())
    
    #Model application
    quarter, smooth, unsmoothed = data_prep(alternative_asset_data, 'Return Commodity - USD Unhedged',)
    ResultViz(quarter, smooth, unsmoothed,'Return Commodity')

    #Autocorrel
    for k in range(1,5):
        print(get_autocorrel(smooth,unsmoothed,k))


    
 
 
