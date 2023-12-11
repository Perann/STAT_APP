import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
alternative_asset_data['Return Commodity - USD Unhedged'] = alternative_asset_data['Commodity - USD Unhedged'].pct_change()
alternative_asset_data['Return Private Equity USD Unhedged'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change()
alternative_asset_data['Return UK Property Direct - USD Unhedged'] = alternative_asset_data['UK Property Direct - USD Unhedged'].pct_change()


serie = alternative_asset_data[['QUARTER','Return Private Equity USD Unhedged']].dropna()

datas_to_unsmooth = serie['Return Private Equity USD Unhedged'].reset_index(drop=True)

date = serie['QUARTER']
T = len(date)


def get_alpha(gamma,phi,datas):
    def function_to_minimize(alpha,data):
        res = 0
        for t in range(2,len(datas)):
            res += (data[t] - gamma*(1-alpha) - (alpha + phi)*data[t-1] - alpha*phi*data[t-2])**2
        return res
    return scipy.optimize.minimize(lambda x : function_to_minimize(x, datas),1/2).x[0]


def get_returns(alpha,ObsReturned):
    TrueReturn = np.zeros(T)
    TrueReturn[0] = ObsReturned[0]
    for i in range(1,T):
        TrueReturn[i] = (ObsReturned[i]-alpha*ObsReturned[i-1])/(1-alpha)
    return TrueReturn

def get_gamma_phi(Returns):
    model = ARIMA(Returns, order=(1, 0, 0))
    results = model.fit()
    coeff = results.params
    return (coeff[0],coeff[1])

gamma0 = 1
phi0 = 1
alpha = get_alpha(gamma0,phi0,datas_to_unsmooth)
performance = get_returns(alpha,datas_to_unsmooth)

gamma = get_gamma_phi(get_returns(get_alpha(gamma0,phi0,datas_to_unsmooth),datas_to_unsmooth))[0]
phi = get_gamma_phi(get_returns(get_alpha(gamma0,phi0,datas_to_unsmooth),datas_to_unsmooth))[1]


while np.abs(gamma-gamma0) >= 10**(-3) and np.abs(phi-phi0) >= 10**(-3):
    gamma0 = gamma
    phi0 = phi
    alpha = get_alpha(gamma,phi,datas_to_unsmooth)
    performance = get_returns(alpha,datas_to_unsmooth)
    gamma = get_gamma_phi(performance)[0]
    phi = get_gamma_phi(performance)[1]



plt.plot(date,datas_to_unsmooth, label = 'Observed Returns')
plt.plot(date,performance, label = 'Unsmoothed Returns')
plt.legend()
plt.xticks(rotation=45)
plt.xticks(date[::6])
plt.show()