#Maximum likelihood estimation
import pandas as pd
import numpy as np 
import scipy as scp

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')

Rt = np.array(alternative_asset_data['Private Equity USD Unhedged'])
mu = np.mean(Rt)

Xt = Rt - mu
est_Xt = np.zeros(len(Xt))
est_Xt[0] = 1
for i in range (1,len(Xt)):
    est_Xt[i] = np.mean(Xt[:i+1])

delta = np.zeros(len(Xt))
for i in range(len(Xt)):
    delta[i] = (Xt[i]-est_Xt[i])**2

print(delta)
