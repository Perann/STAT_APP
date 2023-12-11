import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
for key in alternative_asset_data.keys()[1:]:
    alternative_asset_data['Return ' + key] = alternative_asset_data[key].pct_change()

serie = alternative_asset_data[['QUARTER','Return Infrastructure Debt - USD Unhedged']].dropna()
datas_to_unsmooth = serie['Return Infrastructure Debt - USD Unhedged'].reset_index(drop=True)
date = serie['QUARTER']

def get_alpha(gamma,phi,datas):
    def function_to_minimize(alpha,data):
        res = 0
        for t in range(2,len(datas)):
            res += (data[t] - gamma*(1-alpha) - (alpha + phi)*data[t-1] - alpha*phi*data[t-2])**2
        return res
    return scipy.optimize.minimize(lambda x : function_to_minimize(x, datas),1/2).x[0]
def get_returns(alpha,ObsReturned):
    TrueReturn = np.zeros(len(ObsReturned))
    TrueReturn[0] = ObsReturned[0]
    for i in range(1,len(ObsReturned)):
        TrueReturn[i] = (ObsReturned[i]-alpha*ObsReturned[i-1])/(1-alpha)
    return TrueReturn
def get_gamma_phi(Returns):
    model = ARIMA(Returns, order=(1, 0, 0))
    results = model.fit()
    coeff = results.params
    return (coeff[0],coeff[1])

def AR_model(DataFrame,string, gamma0 = 1,phi0 = 1):
    data_to_join = {}
    serie = DataFrame[['QUARTER',string]].dropna()
    data_to_join['QUARTER'] = serie['QUARTER']
    data_to_join[string] = serie[string].reset_index(drop = True)
    
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
    
    data_to_join['Unsmoothed '  + string] = performance
    data_to_join = pd.DataFrame(data_to_join)

    final_df = pd.merge(DataFrame, data_to_join, on = 'QUARTER', how = 'left')
    return final_df

df = AR_model(alternative_asset_data,'Return Infrastructure Debt - USD Unhedged')
