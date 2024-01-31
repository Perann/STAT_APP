import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from AR_functions import get_alpha, get_returns, get_gamma_phi, AR_model


#Preprocessing
alternative_asset_data = pd.read_excel('EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
for key in alternative_asset_data.keys()[1:]:
    alternative_asset_data['Return ' + key] = alternative_asset_data[key].pct_change()

data_to_analyse = alternative_asset_data[['QUARTER','Return Hedge Fund DJ - USD Unhedged']].dropna()

quarter = data_to_analyse['QUARTER']
smooth = data_to_analyse['Return Hedge Fund DJ - USD Unhedged'].reset_index(drop = True)
unsmoothed = AR_model(smooth)
cumulative_smooth = np.cumsum(smooth)
cumulative_unsmoothed = np.cumsum(unsmoothed)



#Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
fig.suptitle('Yield Analysis, unsmoothed by AR', fontsize=16)

ax1.plot(quarter,smooth, label = 'Observed Returns')
ax1.plot(quarter,unsmoothed, label = 'Unsmoothed Returns')
ax1.legend()
ax1.set_xticks(quarter[::15])
ax1.set_xlabel('Quarters')
ax1.set_ylabel('Returns')

ax2.plot(quarter,cumulative_smooth,label = 'Observed cumulated returns')
ax2.plot(quarter, cumulative_unsmoothed,label  = 'Unsmoothed cumulated returns')
ax2.legend()
ax2.set_xticks(quarter[::15])
ax2.set_xlabel('Quarters')
ax2.set_ylabel('Cumulated returns')

plt.show()

