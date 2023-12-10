import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
classic_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')

plt.plot(alternative_asset_data['QUARTER'], alternative_asset_data['Private Equity USD Unhedged'])
plt.title('Original Time-Series Data')
plt.xlabel('Time')
plt.ylabel('Data')
plt.show()



def get_unsmoothed_return(alpha, rt, rt_1):
    return (rt**alpha - alpha*rt_1**alpha)/(1-alpha)