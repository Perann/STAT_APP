import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from AR_functions import get_alpha, get_returns, get_gamma_phi, AR_model

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
for key in alternative_asset_data.keys()[1:]:
    alternative_asset_data['Return ' + key] = alternative_asset_data[key].pct_change()

print(alternative_asset_data.keys())

data_to_analyse = alternative_asset_data[['QUARTER','Return Hedge Fund DJ - USD Unhedged']].dropna()

quarter = data_to_analyse['QUARTER']
datas_to_unsmooth = data_to_analyse['Return Hedge Fund DJ - USD Unhedged'].reset_index(drop = True)

unsmoothed = AR_model(datas_to_unsmooth)

plt.plot(quarter,datas_to_unsmooth, label = 'Observed Returns')
plt.plot(quarter,unsmoothed, label = 'Unsmoothed Returns')
plt.legend()
plt.title('Returns Hedge Fund DJ - USD Unhedged unsmoothed with AR method')
plt.xticks(quarter[::15])
plt.xlabel('Quarters')
plt.ylabel('Returns')
plt.show()
