# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


from AR_Model.AR_functions import AR_model, get_alpha, get_gamma_phi, get_returns

from getmansky.GetmanskyMain import GetmanskyModel

# Importing the dataset
alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
classic_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

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

getmansky = GetmanskyModel(2)
getmansky.set_default_weights("equal")
getmansky.fit(results['US Equity USD Unhedged'].values.reshape(-1, 1), results['returns PE'].values.reshape(-1,1))
results[f'returns unsmoothed getmansky'] = getmansky.predict(results['US Equity USD Unhedged'])



data = results['returns PE']
UnsmoothedAR = AR_model(data)

plt.plot(results.index.tolist(), results['returns PE'], label = 'Observed Returns')
plt.plot(results.index.tolist(),UnsmoothedAR, label = 'Unsmoothed Returns (AR)')
plt.plot(results.index.tolist(),results['returns unsmoothed getmansky'], label = 'Unsmoothed Returns (Getmansky)')
plt.legend()
plt.title('Comparison unsmoothing AR and Getmansky on Private Equity')
plt.xticks(results.index.tolist()[::15])
plt.xlabel('Quarters')
plt.ylabel('Returns')
plt.show()