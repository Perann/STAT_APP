import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def get_alpha(gamma,phi,datas):
    def function_to_minimize(alpha,data):
        res = 0
        for t in range(2,len(datas)):
            res += (data[t] - gamma*(1-alpha) - (alpha + phi)*data[t-1] + alpha*phi*data[t-2])**2
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

def AR_model(datas_to_unsmooth,gamma0 = 1,phi0= 1):
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
    
    return performance
 
def AR_rebase(serie ,quarter_returns, window = None) -> 'numpy.ndarrray' :
    if  window == None:
        serie = AR_model(serie)
        for k in range(len(quarter_returns)):
            serie[k+2] = (1 + quarter_returns[k])/((1 + serie[k+1])*(1 + serie[k])) - 1
        return serie

if __name__ == '__main__':
   
    alternative_asset_data = pd.read_excel('EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
    for key in alternative_asset_data.keys()[1:]:
        alternative_asset_data['Return ' + key] = alternative_asset_data[key].pct_change()
    data_to_analyse = alternative_asset_data[['QUARTER','Return Hedge Fund DJ - USD Unhedged']].dropna()
    quarter = data_to_analyse['QUARTER']
    datas_to_unsmooth = data_to_analyse['Return Hedge Fund DJ - USD Unhedged'].reset_index(drop = True)
    
    unsmoothed = AR_model(datas_to_unsmooth)
    gamma = 0.014053217162944536
    phi = -0.3169362753345105
    alpha = 0.470059725508072

    a = get_returns(0.4,datas_to_unsmooth)

    test = (unsmoothed[1:]-alpha*unsmoothed[:-1])/(1-alpha)

    Li = [0.02]
    for i in range(1,len(unsmoothed)):
        Li.append((1-alpha)*unsmoothed[i] + alpha*Li[-1])

    plt.plot(quarter,Li,label = 'Unsmoothed Returns')
    plt.show()


