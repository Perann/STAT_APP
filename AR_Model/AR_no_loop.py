import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA



alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
alternative_asset_data['Return Commodity - USD Unhedged'] = alternative_asset_data['Commodity - USD Unhedged'].pct_change()
alternative_asset_data['Return Private Equity USD Unhedged'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change()
alternative_asset_data['Return UK Property Direct - USD Unhedged'] = alternative_asset_data['UK Property Direct - USD Unhedged'].pct_change()


serie = alternative_asset_data[['QUARTER','Return UK Property Direct - USD Unhedged']].dropna()
serie = serie.reset_index(drop=True)

#autoreg
ObservedReturns = serie['Return UK Property Direct - USD Unhedged']
model = ARIMA(ObservedReturns, order=(1, 0, 0))
results = model.fit()
coeff = results.params
gamma,phi = coeff[0],coeff[1]

#find alpha
def get_alpha(gamma,phi,datas):
    def function_to_minimize(alpha,data):
        res = 0
        for t in range(2,len(datas)):
            res += (data[t] - gamma*(1-alpha) - (alpha + phi)*data[t-1] - alpha*phi*data[t-2])**2
        return res
    return scipy.optimize.minimize(lambda x : function_to_minimize(x, datas),1/2).x[0]

opt_alpha = get_alpha(gamma,phi,serie)

# Getting unsmoothed
def get_unsmoothed_return(alpha):
    L = []
    for t in range(1,len(ObservedReturns)):
        L.append(ObservedReturns[t] - alpha*ObservedReturns[t-1]/(1-alpha))
    return L

Unsmoothed_returns = get_unsmoothed_return(opt_alpha)
Observed_Returns = ObservedReturns[1:]


#Plotting
dates  = serie['QUARTER'][1:]
plt.figure()
plt.title('Unsmoothing with AR(1) method')
plt.plot(dates, Observed_Returns, c = 'darkblue', label = 'Observed Returns')
plt.plot(dates, Unsmoothed_returns, 'o--', c = 'orange', ms = 4, label = 'Unsmoothed Returns')
plt.legend()
plt.xticks(rotation=45)
plt.xticks(dates[::4])
plt.show()





