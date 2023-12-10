import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AR_function import get_ar_unsmoothed

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
alternative_asset_data['Return Commodity - USD Unhedged'] = alternative_asset_data['Commodity - USD Unhedged'].pct_change()
alternative_asset_data['Return Private Equity USD Unhedged'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change()
alternative_asset_data['Return UK Property Direct - USD Unhedged'] = alternative_asset_data['UK Property Direct - USD Unhedged'].pct_change()

print(alternative_asset_data.head())

datas_to_unsmooth = alternative_asset_data[['QUARTER','Return Commodity - USD Unhedged']].dropna()
datas_to_unsmooth = datas_to_unsmooth.reset_index(drop = True)

ObservedReturns = datas_to_unsmooth['Return Commodity - USD Unhedged']
unsmoothed = get_ar_unsmoothed(ObservedReturns)
dates  = datas_to_unsmooth['QUARTER'][1:]


plt.figure()
plt.title('Unsmoothing with AR(1) method')
plt.plot(dates, ObservedReturns[1:], c = 'darkblue', label = 'Observed Returns')
plt.plot(dates, unsmoothed, 'o--', c = 'orange', ms = 4, label = 'Unsmoothed Returns')
plt.legend()
plt.xticks(rotation=45)
plt.xticks(dates[::4])
plt.show()

