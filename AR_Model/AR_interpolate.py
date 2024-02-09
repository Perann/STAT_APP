import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

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
    linear_smoohed = []
    for k in range(0,len(datas_to_unsmooth)-1):
        linear_smoohed = linear_smoohed + [datas_to_unsmooth[k],(1/3)*(datas_to_unsmooth[k] + datas_to_unsmooth[k+1]),(2/3)*(datas_to_unsmooth[k] + datas_to_unsmooth[k+1])]

    alpha = get_alpha(gamma0,phi0,linear_smoohed)
    performance = get_returns(alpha,linear_smoohed)
    gamma = get_gamma_phi(get_returns(get_alpha(gamma0,phi0,linear_smoohed),linear_smoohed))[0]
    phi = get_gamma_phi(get_returns(get_alpha(gamma0,phi0,linear_smoohed),linear_smoohed))[1]

    while np.abs(gamma-gamma0) >= 10**(-3) and np.abs(phi-phi0) >= 10**(-3):
        gamma0 = gamma
        phi0 = phi
        alpha = get_alpha(gamma,phi,linear_smoohed)
        performance = get_returns(alpha,linear_smoohed)
        gamma = get_gamma_phi(performance)[0]
        phi = get_gamma_phi(performance)[1]
    return (linear_smoohed,performance)
    

#def datetime_range(start_date, end_date, step):
    #current_date = start_date
    #while current_date <= end_date:
        #yield current_date
        # Add one month to the current date
        #year = current_date.year + ((current_date.month + step - 1) // 12)
        #month = ((current_date.month + step - 1) % 12) + 1
        #current_date = datetime(year, month, current_date.day)


#def quarter_to_datetime(quarter_string):
    #year, quarter = quarter_string.split('-Q')
    #month = 3 * (int(quarter) - 1) + 1  # Assuming quarters start from 1
    #return datetime(int(year), month, 1)




if __name__ == '__main__':
    alternative_asset_data = pd.read_excel('EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
    for key in alternative_asset_data.keys()[1:]:
        alternative_asset_data['Return ' + key] = alternative_asset_data[key].pct_change()
    data_to_analyse = alternative_asset_data[['QUARTER','Return Hedge Fund DJ - USD Unhedged']].dropna()
    quarter = data_to_analyse['QUARTER']
    datas_to_unsmooth = data_to_analyse['Return Hedge Fund DJ - USD Unhedged'].reset_index(drop = True)
    
    smoothed,unsmoothed = AR_model(datas_to_unsmooth)[0], AR_model(datas_to_unsmooth)[1]

    #begin = quarter_to_datetime(data_to_analyse['QUARTER'][0])
    #end = quarter_to_datetime(data_to_analyse['QUARTER'][-1])

    time = np.arange(0,len(smoothed))
    
    plt.plot(time[0:20],smoothed[0:20],'--', label = 'linear smoothed Returns')
    for i in range(0, len(smoothed[0:20]),3):
       plt.plot(time[i], smoothed[i], 'bo')  
    
    plt.plot(time[0:20],unsmoothed[0:20],'r--', label = 'Unsmoothed Returns')
    for i in range(0, len(unsmoothed[0:20])):
        plt.plot(time[i+1], unsmoothed[i+1], 'ro')  
    
    
    plt.legend()
    plt.show()




