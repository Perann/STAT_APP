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

def get_alpha(gamma,phi,datas):
    def function_to_minimize(alpha,data):
        res = 0
        for t in range(2,len(datas)):
            res += (data[t] - gamma*(1-alpha) - (alpha + phi)*data[t-1] - alpha*phi*data[t-2])**2
        return res
    return scipy.optimize.minimize(lambda x : function_to_minimize(x, datas),1/2).x[0]


def det_gamma_phi(gamma,phi,datas):
    alpha = get_alpha(gamma,phi,datas)
    rt_est = [datas[0]]
    for t in range(1,len(datas)):
        rt_est.append(datas[t] - alpha*datas[t-1]/(1-alpha))
    model = ARIMA(rt_est, order=(1, 0, 0))
    results = model.fit()
    coeff = results.params
    return (coeff[0], coeff[1],rt_est)

gamma0 = 1
phi0 = 1
gamma = det_gamma_phi(gamma0,phi0,datas_to_unsmooth)[0]
phi = det_gamma_phi(gamma0,phi0,datas_to_unsmooth)[1]
est = det_gamma_phi(gamma0,phi0,datas_to_unsmooth)[2]


while np.abs(gamma-gamma0) >= 10**(-3) and np.abs(phi-phi0) >= 10**(-3):
    gamma0 = gamma
    phi0 = phi
    est0 = est
    gamma = det_gamma_phi(gamma0,phi0,est0)[0]
    phi = det_gamma_phi(gamma0,phi0,est0)[1]
    est = det_gamma_phi(gamma0,phi0,est0)[2]


date = serie['QUARTER']
T = len(date)
unsmoothed = [datas_to_unsmooth[0]]
for _ in range(T-1):
    unsmoothed.append(gamma + phi*unsmoothed[-1])

plt.plot(date,datas_to_unsmooth)
plt.plot(date,est)
plt.show()
