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
    alternative_asset_data['returns PE'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change()
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data = alternative_asset_data.set_index('QUARTER')

    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('Date', drop = True).resample('Q').last()
    classic_asset_data = classic_asset_data.set_index('QUARTER')
    classic_asset_data = classic_asset_data.pct_change()
    classic_asset_data.dropna(inplace = True)

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True)
    results.drop(columns = ['Private Equity USD Unhedged', 'Liquidity USD Unhedged', 'US Government Bond USD Unhedged'], inplace = True)

    # Getmansky model
    results['returns PE'].plot(label = 'Rt smoothed')
    getmansky = GetmanskyModel(2)
    getmansky.set_default_weights("equal")
    for name in results.columns[:-1]:
        print(name)
        getmansky.fit(results[name].values.reshape(-1, 1), results['returns PE'].values.reshape(-1,1))
        results[f'returns unsmoothed_{name}'] = getmansky.predict(results[name])
        results[f'returns unsmoothed_{name}'].plot(label = f'Rt unsmoothed {name}')
    
    plt.title("Getmansky model with equal weights PE/multibench")
    plt.legend()
    #plt.savefig('getmansky_equal_weights_PE_multibench.png')
    plt.show()