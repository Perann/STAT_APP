"""
In this file we will implement a first version of the AR model.
This model will be used to not only unsmooth the returns of the alternative asset but also to interpolate the missing values.
More precisely, this model will allow us to interpolate monthly returns from quarterly returns.
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys

# Class of the "AR model"
class ARModel:
    def __init__(self, order_theoritical = 1, order_appraised = 1):
        self.order_theoritical = order_theoritical
        self.order_appraised = order_appraised
        self.gamma = 1
        self.phi = 1
        self.alpha = 0.5

    def fit(self, Rto):
        Rto = np.array(Rto)
        _array = []

        def _error_function(x):
            gamma, phi, alpha = x
            Rt = [Rto[0]/3, Rto[0]/3] + [0]*(len(Rto)*3-2)
            Rto_pred = [Rto[0]/3, Rto[0]/3] + [0]*(len(Rto)*3-2)
            for i in range(2, len(Rto)*3):
                Rt[i] = gamma + phi*Rt[i-1]
            for i in range(2, len(Rto)*3):
                Rto_pred[i] = alpha*Rto_pred[i-1] + (1-alpha)*Rt[i-1]
            Rto_pred_comp = np.array([(1+Rto_pred[i*3])*(1+Rto_pred[i*3+1])*(1+Rto_pred[i*3+2])-1 for i in range(len(Rto))])
            _array.append(Rto_pred_comp)
            return np.sum((Rto-Rto_pred_comp)**2)
        
        opti = scipy.optimize.minimize(_error_function, [0.4, 0.5, 0.6],
                                        bounds = ((None, None), (None, None), (0, 1)))
                                        #options = {'maxiter': 1})
        print(opti.x)
        print("tableau", _array)
        self.gamma, self.phi, self.alpha = opti.x[0], opti.x[1], opti.x[2]

    def predict(self, Rto):
        Rto = np.array(Rto)
        Rto_pred = [Rto[0]/3, Rto[0]/3]
        for i in range(2, len(Rto)*3):
            Rto_pred.append(self.gamma*(1-self.alpha) + (self.alpha+self.phi)*Rto_pred[i-1] - self.alpha*self.phi*Rto_pred[i-2])
        return np.array(Rto_pred)




if __name__ == "__main__":
    # Importing the dataset
    alternative_asset_data = pd.read_excel('/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
    classic_asset_data = pd.read_excel('/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

    # Preprocessing
    alternative_asset_data = alternative_asset_data[['QUARTER', 'Private Equity USD Unhedged']]
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data['returns PE'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change(fill_method=None)
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data = alternative_asset_data.set_index('QUARTER')

    classic_asset_data = classic_asset_data[['QUARTER', 'Date', 'US Equity USD Unhedged']]
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('Date', drop = False).resample('M').last()
    classic_asset_data.dropna(inplace = True) #to deal with the problem of february
    classic_asset_data['returns US equity'] = classic_asset_data['US Equity USD Unhedged'].pct_change(fill_method=None)
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('QUARTER')

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True).drop(columns = ['US Equity USD Unhedged', 'Private Equity USD Unhedged'])
    results = results[1:]

    results = results.set_index('Date')

    for line in results.iterrows():
        if line[0].month in [1, 2, 4, 5, 7, 8, 10, 11]:
            line[1]['returns PE'] = None

    end_date_forced = '31-12-2022' 
    results = results[:end_date_forced]

    results_ = results.dropna()
    #print(results_['returns PE'].values.reshape(-1,1))

    ar_model = ARModel(order_theoritical = 1, order_appraised = 1)
    ar_model.fit(results_['returns PE'].values.reshape(-1,1))
    #ar_model.phi, ar_model.alpha, ar_model.gamma = 0.005, 0.5, 0.005
    results['returns unsmoothed'] = ar_model.predict(results_['returns PE'].values.reshape(-1,1))

    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    results['returns PE TR'] = (results['returns PE']+1).cumprod()-1
    results_no_interpolation = results.resample('Q').last() #just to view the trend

    # Restricting the dates
    start_date = '2006-08-31'
    end_date = '2010-09-30'
    results_sliced = results.loc[start_date:end_date]
    #print(results)

    # Plotting
    # define subplot layout
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].title.set_text("Getmansky model with eq weights PE/US equity")

    results_no_interpolation['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed', ax=axes[0])
    results_no_interpolation['returns PE TR'].plot(label = 'Rt PE', ax=axes[0])

    results['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed', marker = 'o', linestyle = '', ax=axes[1])
    results['returns PE TR'].plot(label = 'Rt PE', marker = 'o', linestyle = '', ax=axes[1])

    results_sliced['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed', marker = 'o', linestyle = '', ax=axes[2])
    results_sliced['returns PE TR'].plot(label = 'Rt PE', marker = 'o', linestyle = '', ax=axes[2])

    plt.legend()
    #plt.savefig(f'getmansky/output/GetmanskyPres/GetmanskyModel_eqweight_{2}_PE_best.png')
    plt.show()