import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AR_functions import AR_model

def quarter_rebase(data : 'numpy.ndarray', quarter_return : 'float64') -> 'numpy.ndarray':
    data[-1] = (1 + quarter_return)/(1 + data[-2])*(1 + data[-3]) - 1 
    return data[-1]

alternative_data_raw = pd.read_excel('EnsaeAlternativeTimeSeries.xlsx', sheet_name='Alternative Asset', index_col=0)

def tweak_alternative_data(df_):
    return (df_.assign(**{col + '_%y/y': df_[col].pct_change(fill_method = None) for col in df_.columns}) #Adding the returs
            .set_index(pd.to_datetime(df_.index)) 
            .rename(lambda c: c.replace(' ','_'), axis = 1))


quarter = (tweak_alternative_data(alternative_data_raw)['Global_Property_USD_Unhedged']
        .pct_change(fill_method=None)
        .dropna()
        .values)



interpol2 = (tweak_alternative_data(alternative_data_raw)['Global_Property_USD_Unhedged']
            .resample('MS')
            .interpolate(method = 'polynomial', order = 2)
            .pct_change(fill_method=None)
            .dropna()
            .values)

res = np.zeros(len(quarter))
for k in range(len(quarter)):
    res[k:k+3] = AR_model(interpol2[k:k+3])
    res[k+2] = quarter_rebase(res[k:k+3],quarter[k])
print(res)



