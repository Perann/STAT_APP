import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')

PvEquity = alternative_asset_data['Private Equity USD Unhedged'].pct_change()


#autoregression:

model = ARIMA(PvEquity, order=(1, 0, 0))
results = model.fit()
predictions = results.predict(start=1, end=len(PvEquity)-1)
coeff = results.params

gamma = coeff[0]
phi = coeff[1]

#F alpha

def function_to_minimize(alpha):
    res = 0
    for t in range(2,len(PvEquity)):
        res += ((PvEquity[t]) - gamma*(1-alpha) - (alpha+phi)*(PvEquity[t-1]) - alpha*phi*PvEquity[t-2])**2
    return res


alpha_final = scipy.optimize.minimize(function_to_minimize, 0, method='L-BFGS-B').x[0]





def get_unsmoothed_return(alpha,r):
    L = []
    for t in range(1,len(PvEquity)):
        L.append(r[t] - alpha*(r[t-1])/(1-alpha))
    return L





OriginalReturn = PvEquity
UnsmoothedReturn = get_unsmoothed_return(alpha_final,PvEquity)




plt.plot(alternative_asset_data['QUARTER'][:], OriginalReturn)
plt.plot(alternative_asset_data['QUARTER'][1:], UnsmoothedReturn)
plt.title('Original and unsmooth returns')
plt.xlabel('Time')
plt.ylabel('Data')
plt.show()
