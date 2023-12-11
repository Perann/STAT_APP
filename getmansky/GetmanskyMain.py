"""
In this file we will implement a first version of the getmansky model.
Espacially we will implement the three classical type of weights :
- Equal weights
- Sum of years (linearly decreasing with time)
- Geometric (exponentialy decreasing with time)
"""

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

# Importing the dataset
alternative_asset_data = pd.read_excel('/Users/adam/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
classic_asset_data = pd.read_excel('/Users/adam/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

# Preprocessing
alternative_asset_data = alternative_asset_data[['QUARTER', 'Hedge Fund DJ - USD Unhedged']]
alternative_asset_data.dropna(inplace = True)
alternative_asset_data['returns hedge fund'] = alternative_asset_data['Hedge Fund DJ - USD Unhedged'].pct_change()
alternative_asset_data.dropna(inplace = True)
alternative_asset_data = alternative_asset_data.set_index('QUARTER')

classic_asset_data = classic_asset_data[['QUARTER', 'Date', 'US Equity USD Unhedged']]
classic_asset_data.dropna(inplace = True)
classic_asset_data = classic_asset_data.set_index('Date', drop = False).resample('Q').last()
classic_asset_data['returns US equity'] = classic_asset_data['US Equity USD Unhedged'].pct_change()
classic_asset_data = classic_asset_data.set_index('QUARTER')
classic_asset_data.dropna(inplace = True)

results = classic_asset_data.copy()
results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True).drop(columns = ['Date', 'US Equity USD Unhedged', 'Hedge Fund DJ - USD Unhedged'])
result = results.copy() # à enlever
#Compute the real returns
type_ = 'equal'
k = 2
mu = np.mean(results['returns hedge fund'])
weights = Weights(type_, k)

_tmp = [results['returns hedge fund'].iloc[0], results['returns hedge fund'].iloc[1]]

for i in range(2, len(results)):
    _tmp.append((results['returns hedge fund'].iloc[i] - np.dot(weights.list[1:], _tmp[-2:]))/weights.list[0])

_tmp[0], _tmp[1] = np.nan, np.nan
results['Rt'] = _tmp
results.dropna(inplace = True)

lr = LinearRegression()
lr.fit(results['returns US equity'].values.reshape(-1, 1), results['Rt'].values.reshape(-1, 1))
beta, mu = lr.coef_[0, 0], lr.intercept_[0]

results['returns R'] = mu + beta*results['returns US equity']
print(results)

# results['returns R'].plot(label = 'Rt unsmoothed')
# results['returns hedge fund'].plot(label = 'Rt smoothed')
# plt.legend()
# #plt.savefig(f'germansky/output/getmanskyModel/GetmanskyModel_True_{type_}_{k}.png')
# plt.show()


class GetmanskyModel:
    def __init__(self, k):
        self.k = k
        self.weights = Weights("equal", k)
        self.mu = 0
        self.beta = 1

    def set_default_weights(self, type_, delta = None):
        self.weights = Weights(type_, self.k, delta)

    def optimize_weights(self):
        
        pass

    def fit(self, Benchmark, Rto):
        assert len(Rto) == len(Benchmark), f"Rto and Benchmark must have the same length \n The length of Rto is {len(Rto)} \n The length of Benchmark is {len(Benchmark)}"
        _tmp = [Rto[0], Rto[1]]
        for i in range(2, len(Rto)):
            _tmp.append((Rto[i] - np.dot(self.weights.list[1:], _tmp[-self.k:]))/self.weights.list[0])
        Benchmark, _tmp = np.array(Benchmark[2:]), np.array(_tmp[2:])

        lr = LinearRegression()
        lr.fit(Benchmark, _tmp)

        self.beta, self.mu = lr.coef_[0, 0], lr.intercept_[0]

    def predict(self, Benchmark):
        return self.mu + self.beta*np.array(Benchmark)


getmansky = GetmanskyModel(2)
getmansky.set_default_weights('equal')
getmansky.fit(result['returns US equity'].values.reshape(-1, 1), result['returns hedge fund'].values.reshape(-1,1))
results['returns R2'] = getmansky.predict(results['returns US equity'])
print(results)