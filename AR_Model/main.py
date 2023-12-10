import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from autoreg_coeff import auto_reg

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
classic_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

alternative_asset_data['Return Commodity - USD Unhedged'] = alternative_asset_data['Commodity - USD Unhedged'].pct_change()
alternative_asset_data['Return Private Equity USD Unhedged'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change()
alternative_asset_data['Return UK Property Direct - USD Unhedged'] = alternative_asset_data['UK Property Direct - USD Unhedged'].pct_change()


serie = alternative_asset_data[['QUARTER','Return UK Property Direct - USD Unhedged']].dropna()
serie = serie.reset_index(drop=True)

#autoreg
ObservedReturns = serie['Return UK Property Direct - USD Unhedged']
gamma,phi = auto_reg(ObservedReturns)

#find alpha
def function(alpha):
    res = 0
    for t in range(2,len(ObservedReturns)):
        res += (ObservedReturns[t] - gamma*(1-alpha) -(alpha + phi)*ObservedReturns[t-1] - alpha*phi*ObservedReturns[t-2])**2
    return res

a = scipy.optimize.minimize(function,1/2).x[0]

# Getting unsmoothed
def get_unsmoothed_return(alpha):
    L = []
    for t in range(1,len(ObservedReturns)):
        L.append(ObservedReturns[t] - alpha*ObservedReturns[t-1]/(1-alpha))
    return L



#Plotting
Observed_Returns = ObservedReturns[1:]
dates  = serie['QUARTER'][1:]
Unsmoothed_returns = get_unsmoothed_return(a)
plt.figure()
plt.title('Unsmoothing with AR(1) method')
plt.plot(dates, Observed_Returns, c = 'darkblue', label = 'Observed Returns')
plt.plot(dates, Unsmoothed_returns, 'o--', c = 'orange', ms = 4, label = 'Unsmoothed Returns')
plt.legend()
plt.xticks(rotation=45)
plt.xticks(dates[::4])
plt.show()





