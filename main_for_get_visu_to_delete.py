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
    classic_asset_data = classic_asset_data[:299]
    classic_asset_data['returns US equity'] = classic_asset_data['US Equity USD Unhedged'].pct_change(fill_method=None)
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('QUARTER')

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True).drop(columns = ['US Equity USD Unhedged', 'Private Equity USD Unhedged'])
    results = results[1:]

    getmansky = GetmanskyModel(2)
    getmansky.set_default_weights("equal")
    getmansky.fit(results['returns US equity'].values.reshape(-1, 1), results['returns PE'].values.reshape(-1,1))
    results['returns unsmoothed'] = getmansky.predict(results['returns US equity'])

    results = results.set_index('Date')

    for i in range(len(results['returns PE'])):
        if (i%3 == 0 or i%3 == 1):
            results['returns PE'].iloc[i] = None

    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    results['returns PE TR'] = (results['returns PE']+1).cumprod()-1

    #to check
    print("nb points PE : ", len(results['returns PE'].dropna()))
    print("nb points unsmoothed : ", len(results['returns unsmoothed TR']))

    #restricting
    start_date = '2006-08-31'
    end_date = '2010-09-30'
    results = results.loc[start_date:end_date]

    # plotting
    results['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed', marker = 'o', linestyle = '')
    results['returns PE TR'].plot(label = 'Rt PE', marker = 'o', linestyle = '')
    plt.title("Getmansky model with reglin weights PE/US equity")
    plt.legend()
    #plt.savefig(f'getmansky/output/GetmanskyPres/GetmanskyModel_eqweight_{2}_PE_unsmooth_restricted.png')
    plt.show()