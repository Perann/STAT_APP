import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')

PvEquity = alternative_asset_data['Private Equity USD Unhedged']


#autoregression:

model = ARIMA(PvEquity, order=(1, 0, 0))
results = model.fit()
predictions = results.predict(start=1, end=len(PvEquity)-1)
coeff = results.params

gamma = coeff[0]
phi = coeff[1]



def function_to_minimize(alpha):
    res = 0
    for t in range(2,len(PvEquity)):
        res += (((PvEquity[t]**alpha) - gamma*(1-alpha) - (alpha+phi)*(PvEquity[t-1]**alpha) - alpha*phi*PvEquity[t-2])**2)**2
    return res

tolerance = 10**(-3)
alpha_final = scipy.optimize.minimize(function_to_minimize, 0, options={'tol': tolerance})

print(alpha_final)