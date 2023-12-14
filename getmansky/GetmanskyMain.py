"""
In this file we will implement a first version of the getmansky model.
Espacially we will implement the three classical type of weights :
- Equal weights
- Sum of years (linearly decreasing with time)
- Geometric (exponentialy decreasing with time)
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

# Importing packages from the prject
sys.path.append("getmansky/")
from WeightsFunctions.weights import Weights


# Class of the Getmansky model
class GetmanskyModel:
    def __init__(self, k):
        self.k = k
        self.weights = Weights("equal", k)
        self.mu = 0
        self.beta = 1

    def set_default_weights(self, type_, delta = None):
        self.weights = Weights(type_, self.k, delta)

    def optimize_weights_MLE(self, Rto):
        # Xt = Rto - np.mean(Rto)
        # n = len(Rto)
        # def S(theta):
        # return 1/n * np.sum([((Xt-np.mean(Xt[:i]))**2)/  )
        # return Xt
        pass

    def optimize_weights_LR(self, Benchmark, Rto):
        df = pd.DataFrame([Benchmark, Rto], index = ['Benchmark', 'Rto']).T
        for i in range(1, self.k+1):
            df[f'bench_lag_{i}'] = df['Benchmark'].shift(i)
        df.dropna(inplace = True)
        X, y = df.drop(columns = ['Rto']), df['Rto']
        lr = LinearRegression()
        lr.fit(X, y)
        self.weights.list = lr.coef_/np.sum(lr.coef_) #careful with the order of thetas

    def fit(self, Benchmark, Rto):
        Benchmark, Rto = np.array(Benchmark), np.array(Rto)
        Rto = [Rto[i*3] for i in range(len(Rto)//3)]

        def error_function(x):
            beta, mu = x
            Rt = mu + beta*Benchmark
            Rto_pred = [Rt[0], Rt[1], Rt[2]]
            for _ in range(3, len(Rt)):
                Rto_pred.append(np.dot(self.weights.list, Rto_pred[:-(self.k+2):-1]))
            Rto_comp = np.array([(1+Rto_pred[i*3])*(1+Rto_pred[i*3+1])*(1+Rto_pred[i*3+2])-1 for i in range(len(Rto_pred)//3)])
            return np.sum((Rto - Rto_comp)**2)
        
        opti = scipy.optimize.minimize(error_function, [0.5, 1])
        self.beta, self.mu = opti.x[0], opti.x[1]

    def predict(self, Benchmark):
        return self.mu + self.beta*np.array(Benchmark)



if __name__ == "__main__":
    # Importing the dataset
    alternative_asset_data = pd.read_excel('/Users/adam/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
    classic_asset_data = pd.read_excel('/Users/adam/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')

    # Preprocessing
    alternative_asset_data = alternative_asset_data[['QUARTER', 'Private Equity USD Unhedged']]
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data['returns PE'] = alternative_asset_data['Private Equity USD Unhedged'].pct_change()
    alternative_asset_data.dropna(inplace = True)
    alternative_asset_data = alternative_asset_data.set_index('QUARTER')

    classic_asset_data = classic_asset_data[['QUARTER', 'Date', 'US Equity USD Unhedged']]
    classic_asset_data.dropna(inplace = True)
    classic_asset_data = classic_asset_data.set_index('Date', drop = False).resample('M').last()
    classic_asset_data['returns US equity'] = classic_asset_data['US Equity USD Unhedged'].pct_change()
    classic_asset_data = classic_asset_data.set_index('QUARTER')
    classic_asset_data.dropna(inplace = True)

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'outer', left_index = True, right_index = True).drop(columns = ['US Equity USD Unhedged', 'Private Equity USD Unhedged'])
    results.dropna(inplace=True)
    results = results[1:]

    getmansky = GetmanskyModel(2)
    getmansky.set_default_weights("equal")
    getmansky.fit(results['returns US equity'].values.reshape(-1, 1), results['returns PE'].values.reshape(-1,1))
    results['returns unsmoothed'] = getmansky.predict(results['returns US equity'])

    results = results.set_index('Date')

    results['returns unsmoothed final'] = 0

    for i in range(2, len(results['returns unsmoothed'])):
        results['returns unsmoothed final'].iloc[i] = (results['returns unsmoothed'].iloc[i-2]+1)*(results['returns unsmoothed'].iloc[i-1]+1)*(results['returns unsmoothed'].iloc[i]+1)-1
    
    #results = results.resample('Q').last()
    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    results = results.resample('Q').last()
    results['returns PE TR'] = (results['returns PE']+1).cumprod()-1
    print(results.head(50))
    results['returns unsmoothed TR'].plot(label = 'Rt unsmoothed TR')
    results['returns PE TR'].plot(label = 'Rt smoothed')
    plt.title("Getmansky model with reglin weights PE/US equity")
    plt.legend()
    #plt.savefig(f'getmansky/output/GetmanskyPres/GetmanskyModel_reglin_{k}_PE.png')
    plt.show()

    # import statsmodels.api as sm
    # data = sm.datasets.macrodata.load_pandas()
    # rgdpg = data.data['realgdp'].pct_change().dropna()
    # acov = sm.tsa.acovf(rgdpg, fft = False, nlag = 2)
    # theta, sigma2  = sm.tsa.stattools.innovations_algo(acov)
    # print(acov)
    # print(theta)