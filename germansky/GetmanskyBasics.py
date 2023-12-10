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

# Compute the real returns
type_ = 'sumOfYears'
k = 2
mu = np.mean(results['returns hedge fund'])
weights = Weights(type_, k)

def objective_function(beta):
    """This function computes the objective function of the Getmansky model"""
    
    _gen = (i for i in results['returns US equity'].rolling(3) if len(i) == 3)
    _sum = np.array([np.dot(i.values,weights.list) for i in _gen])

    return np.sum((results['returns hedge fund'][2:]-mu-beta*_sum)**2)

beta = scipy.optimize.minimize(objective_function, x0 = 1).x[0]

results['returns R'] = mu + beta*results['returns US equity']
results['returns R'].plot(legend = 'Rt')
results['returns hedge fund'].plot(legend = 'Rto')
plt.legend()
plt.savefig(f'germansky/output/getmanskyModel/GetmanskyModel_{type_}_{k}.png')
plt.show()
