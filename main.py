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
    alternative_asset_data = alternative_asset_data[['QUARTER', 'Private Equity USD Unhedged']]
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data['returns PE'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change(fill_method=None)
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data = alternative_asset_data.set_index('QUARTER')

    classic_asset_data = classic_asset_data[['QUARTER', 'Date', 'US Equity USD Unhedged']]
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('Date', drop = False).resample('M').last()
    classic_asset_data.dropna(inplace = True) #to deal with the problem of february
    classic_asset_data['returns US equity'] = classic_asset_data['US Equity USD Unhedged'].pct_change(fill_method=None)
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('QUARTER')

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True).drop(columns = ['US Equity USD Unhedged', 'Private Equity USD Unhedged'])
    results = results[1:]

    getmansky = GetmanskyModel(2)
    #getmansky.set_default_weights("equal")
    getmansky.optimize_weights_LR(results['returns US equity'].values, results['returns PE'].values)
    getmansky.fit(results['returns US equity'].values.reshape(-1, 1), results['returns PE'].values.reshape(-1,1))
    results['returns unsmoothed'] = getmansky.predict(results['returns US equity'])

    results = results.set_index('Date')

    for line in results.iterrows():
        if line[0].month in [1, 2, 4, 5, 7, 8, 10, 11]:
            line[1]['returns PE'] = None

    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    results['returns PE TR'] = (results['returns PE']+1).cumprod()-1
    results_no_interpolation = results.resample('Q').last() #just to view the trend

    # Restricting the dates
    end_date_forced = '30-06-2023' #just for the visualisation
    results = results[:end_date_forced]
    results_no_interpolation = results_no_interpolation[:end_date_forced]

    start_date = '2006-08-31'
    end_date = '2010-09-30'
    results_sliced = results.loc[start_date:end_date]


    # Autocorrelation and volatility
    results['vol rolling 1y Rt unsmoothed'] = results['returns unsmoothed'].rolling(window = 12).std()*np.sqrt(12)*100 # in %
    results['vol rolling 1y Rt PE'] = results['returns PE'].dropna().rolling(window = 4).std()*np.sqrt(4)*100 # in %

    results['vol rolling 5y Rt unsmoothed'] = results['returns unsmoothed'].rolling(window = 12*5).std()*np.sqrt(12)*100 # in %
    results['vol rolling 5y Rt PE'] = results['returns PE'].dropna().rolling(window = 4*5).std()*np.sqrt(4)*100 # in %
    
    results['vol rolling 10y Rt unsmoothed'] = results['returns unsmoothed'].rolling(window = 12*10).std()*np.sqrt(12)*100 # in %
    results['vol rolling 10y Rt PE'] = results['returns PE'].dropna().rolling(window = 4*10).std()*np.sqrt(4)*100 # in %

    auto_corr = [results['returns unsmoothed TR'].autocorr(i) for i in range(50)]

    results = results.dropna()
    
    # Plotting
    # define subplot layout
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('Getmansky model PE on US equity volatility 10y', fontsize=12)
    
    axes.set_ylabel('volatility (%)')

    results['vol rolling 10y Rt unsmoothed'].plot(label = 'volatility PE unsmoothed')
    results['vol rolling 10y Rt PE'].plot(label = 'volatility PE')


    plt.legend()
    #plt.savefig(f'getmansky/output/GetmanskyPres_8_fev/GetmanskyModelvolatility_rolling_10y_LR_weights_{2}_PE_US_equity.png')
    plt.show()