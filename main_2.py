# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing packages from the prject
#from getmansky.WeightsFunctions.weights import Weights
from getmansky.GetmanskyMain import GetmanskyModel


if __name__ == "__main__":
    # Importing the dataset
    alternative_asset_data = pd.read_excel('/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
    classic_asset_data = pd.read_excel('/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

    # Preprocessing
    alternative_asset_data = alternative_asset_data[[a for a in list(alternative_asset_data.keys()) if a != 'Infrastructure Equity Listed - USD Unhedged']]
    alternative_asset_data.dropna(inplace = True)
    list_key_return = []
    for key in alternative_asset_data.keys():
        if key != 'QUARTER':
            alternative_asset_data[f'returns {key}'] = alternative_asset_data[key].pct_change(fill_method=None)
            list_key_return.append(f'returns {key}')
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data = alternative_asset_data.set_index('QUARTER')
    alternative_asset_data = alternative_asset_data[list_key_return]
    #corr = alternative_asset_data.corr()
    #corr.to_excel('correlation_no_interpolation.xlsx')

    classic_asset_data = classic_asset_data[['QUARTER', 'Date', 'US Equity USD Unhedged']]
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('Date', drop = False).resample('M').last()
    classic_asset_data.dropna(inplace = True) #to deal with the problem of february
    classic_asset_data['returns US equity'] = classic_asset_data['US Equity USD Unhedged'].pct_change(fill_method=None)
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('QUARTER')

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True).drop(columns = ['US Equity USD Unhedged'])
    results = results[1:]
    #print(results)

    getmansky = GetmanskyModel(2)
    #getmansky.set_default_weights("equal")
    getmansky.optimize_weights_LR(results['returns US equity'].values, results['returns Private Equity USD Unhedged'].values)
    getmansky.fit(results['returns US equity'].values.reshape(-1, 1), results['returns Private Equity USD Unhedged'].values.reshape(-1,1))
    results['returns unsmoothed'] = getmansky.predict(results['returns US equity'])

    results = results.set_index('Date')

    for line in results.iterrows():
        if line[0].month in [1, 2, 4, 5, 7, 8, 10, 11]:
            line[1]['returns Private Equity USD Unhedged'] = None

    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    results['returns Private Equity USD Unhedged TR'] = (results['returns Private Equity USD Unhedged']+1).cumprod()-1
    results_no_interpolation = results.resample('Q').last() #just to view the trend

    results_no_interpolation.drop(columns=['returns unsmoothed TR', 'returns Private Equity USD Unhedged TR', 'returns US equity'], inplace=True)

    corr = results_no_interpolation.corr()
    corr.to_excel('correlation_interpolation_get.xlsx')
    print(results_no_interpolation)


    # # Restricting the dates
    # end_date_forced = '30-06-2023' #just for the visualisation
    # results = results[:end_date_forced]
    # results_no_interpolation = results_no_interpolation[:end_date_forced]

    # start_date = '2006-08-31'
    # end_date = '2010-09-30'
    # results_sliced = results.loc[start_date:end_date]


    
    # # Plotting
    # # define subplot layout
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # fig.suptitle('Getmansky model PE on US equity volatility 10y', fontsize=12)
    
    # axes.set_ylabel('volatility (%)')

    # results['vol rolling 10y Rt unsmoothed'].plot(label = 'volatility PE unsmoothed')
    # results['vol rolling 10y Rt PE'].plot(label = 'volatility PE')


    # plt.legend()
    # #plt.savefig(f'getmansky/output/GetmanskyPres_8_fev/GetmanskyModelvolatility_rolling_10y_LR_weights_{2}_PE_US_equity.png')
    # plt.show()