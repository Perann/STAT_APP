import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def graph_plot(X:np.array,Y:np.matrix, figsize:tuple = (6,6)):
    
    plt.figure(figsize = figsize)
    title = ''
    for y in Y:
        plt.plot(X,y)
        title = title + ' ' + y.keys()
    plt.xlabel('Dates')
    plt.ylabel('Returns')
    plt.title('title')
    plt.show()


alternative_asset_data = pd.read_excel('C:\\Users\\LENOVO\\Desktop\\EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
alternative_asset_data['Return Commodity - USD Unhedged'] = alternative_asset_data['Commodity - USD Unhedged'].pct_change()
alternative_asset_data['Return Private Equity USD Unhedged'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change()
alternative_asset_data['Return UK Property Direct - USD Unhedged'] = alternative_asset_data['UK Property Direct - USD Unhedged'].pct_change()


serie = alternative_asset_data[['QUARTER','Return Private Equity USD Unhedged']].dropna()

datas_to_unsmooth = serie['Return Private Equity USD Unhedged'].reset_index(drop=True)

date = serie['QUARTER']

graph_plot(date,datas_to_unsmooth)