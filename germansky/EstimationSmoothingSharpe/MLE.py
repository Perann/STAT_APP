#Maximum likelihood estimation
import pandas as pd
import numpy as np 
import scipy as scp

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')

Rt = np.array(alternative_asset_data['Private Equity USD Unhedged'])
mu = np.mean(Rt)
print(mu)
Xt = Rt - mu
print(Xt)