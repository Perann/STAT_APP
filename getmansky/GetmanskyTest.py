# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from sklearn.linear_model import LinearRegression

# Importing packages from the prject
from WeightsFunctions.weights import Weights
from AR_bernousse.ARmain import GetmanskyModel

# Importing the dataset
alternative_asset_data = pd.read_excel('/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
classic_asset_data = pd.read_excel('/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

# Preprocessing
alternative_asset_data = alternative_asset_data[['QUARTER', 'Global Property USD Unhedged']]
alternative_asset_data.dropna(inplace = True)
alternative_asset_data['returns global property'] = alternative_asset_data['Global Property USD Unhedged'].pct_change()
alternative_asset_data.dropna(inplace = True)
alternative_asset_data.drop(columns = ['Global Property USD Unhedged'], inplace = True)
alternative_asset_data = alternative_asset_data.set_index('QUARTER')

classic_asset_data.dropna(inplace = True)
classic_asset_data = classic_asset_data.set_index('Date', drop = True).resample('Q').last()
classic_asset_data = classic_asset_data.set_index('QUARTER')
classic_asset_data = classic_asset_data.pct_change()
classic_asset_data.dropna(inplace = True)

results = classic_asset_data.copy()
results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True)

def objective(weights_):
    _tmp = results.copy()
    _tmp['returns bench'] = weights_[0]*_tmp['Liquidity USD Unhedged'] + weights_[1]*_tmp['US Equity USD Unhedged']+ weights_[2]*_tmp['US Government Bond USD Unhedged']+ weights_[3]*_tmp['USD Corporate Bond - USD Unhedged']
    return -np.corrcoef(_tmp['returns bench'], _tmp['returns global property'])[0,1]

opti = scipy.optimize.minimize(objective, np.array([0.25, 0.25, 0.25, 0.25]), bounds = ((0,1), (0,1), (0,1), (0,1)), constraints = {'type': 'eq', 'fun': lambda x:  np.sum(x)-1})
weights_ = opti.x
print(weights_)
print(opti.success)

results['returns bench'] = weights_[0]*results['Liquidity USD Unhedged'] + weights_[1]*results['US Equity USD Unhedged']+ weights_[2]*results['US Government Bond USD Unhedged']+ weights_[3]*results['USD Corporate Bond - USD Unhedged']
results = results[['returns global property', 'returns bench']]
results.dropna(inplace = True)
k = 2
getmansky = GetmanskyModel(2)
getmansky.optimize_weights_LR(results['returns bench'], results['returns global property'])
getmansky.fit(results['returns bench'].values.reshape(-1, 1), results['returns global property'].values.reshape(-1,1))
results['returns unsmoothed'] = getmansky.predict(results['returns bench'])

results['returns unsmoothed'].plot(label = 'Rt unsmoothed')
results['returns global property'].plot(label = 'Rt smoothed')
plt.title("Getmansky model with reglin weights global property/custom bench")
plt.legend()
#plt.savefig(f'getmansky/output/getmanskyModel/GetmanskyModel_reglin_{k}_globalpropertyVScustombench.png')
plt.show()