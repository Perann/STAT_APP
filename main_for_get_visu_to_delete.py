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

    #print("alter data", alternative_asset_data)

    classic_asset_data = classic_asset_data[['QUARTER', 'Date', 'US Equity USD Unhedged']]
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('Date', drop = False).resample('M').last()
    classic_asset_data['returns US equity'] = classic_asset_data['US Equity USD Unhedged'].pct_change(fill_method=None)
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('QUARTER')

    #print("classic data", classic_asset_data)

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True).drop(columns = ['US Equity USD Unhedged', 'Private Equity USD Unhedged'])
    results = results[1:]
    print("2", results)

    #results['returns PE'] = results[['returns PE']].groupby(['QUARTER']).last()
    print(results)

    #print("resample", results)

    getmansky = GetmanskyModel(2)
    getmansky.set_default_weights("equal")
    getmansky.fit(results['returns US equity'].values.reshape(-1, 1), results['returns PE'].values.reshape(-1,1))
    results['returns unsmoothed'] = getmansky.predict(results['returns US equity'])

    results = results.set_index('Date')

    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1

    #print(results)


    # results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    # results = results.resample('Q').last()
    # results['returns PE TR'] = (results['returns PE']+1).cumprod()-1

    # # plotting
    # results['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed')
    # results['returns PE TR'].plot(label = 'Rt PE')
    # plt.title("Getmansky model with reglin weights PE/US equity")
    # plt.legend()
    # #plt.savefig(f'getmansky/output/GetmanskyPres/GetmanskyModel_eqweight_{2}_PE_best.png')
    # plt.show()