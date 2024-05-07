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
from types import NoneType

# Importing packages from the project
#sys.path.append("getmansky/")
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
        self.weights.list = lr.coef_/np.sum(lr.coef_) #careful with the order of thetas (seems to be ok cf. doc)

    def fit(self, Benchmark, Rto, window=None):
        if not isinstance(window, (int, NoneType)):
            passed = type(window).__name__
            raise TypeError(f"The window argument should be of type int or NoneType (in order to use only 1 window). Here it is " + passed + '.')
        elif isinstance(window, int) and window <= 0:
            passed = str(window)
            raise ValueError(f"You must pass a positive integer as window argument. Here you passed " + passed + '.')

        def _error_function(x, Benchmark, Rto):
            beta, mu = x
            Rt = mu + beta*Benchmark
            Rto_pred = [Rt[0], Rt[1], Rt[2]] + [0]*(len(Rt)-3)
            for i in range(3, len(Rto_pred)):
                Rto_pred[i] = np.dot(self.weights.list, np.array(Rt[i-2:i+1])[::-1]) #ok with the order of the weights (checked)
            Rto_comp = np.array([(1+Rto_pred[i*3])*(1+Rto_pred[i*3+1])*(1+Rto_pred[i*3+2])-1 for i in range(len(Rt)//3)])
            return np.sum((Rto - Rto_comp)**2)
        
        if window is None:
            Benchmark, Rto = np.array(Benchmark), np.array(Rto)
            Rto = np.array([Rto[i*3] for i in range(len(Rto)//3)]) #because we broadcasted during the preprocessing...

            opti = scipy.optimize.minimize(
                fun=_error_function, 
                x0=[0.5, 1],
                args=(Benchmark, Rto)
                )
            self.beta, self.mu = opti.x[0], opti.x[1]
        else :
            self.beta, self.mu = [], []
            Benchmark, Rto = pd.Series(Benchmark.reshape(len(Benchmark))), pd.Series(Rto.reshape(len(Rto)))
            for i, (Benchmark_, Rto_) in enumerate(zip(Benchmark.rolling(window), Rto.rolling(window))):
                if i >= window:
                    if Rto_.iloc[0] == Rto_.iloc[1] and Rto_.iloc[1] == Rto_.iloc[2]:
                        Benchmark_, Rto_ = np.array(Benchmark_), np.array(Rto_)                        
                    elif Rto_.iloc[0] == Rto_.iloc[1] and Rto_.iloc[1] != Rto_.iloc[2]:
                        Benchmark_, Rto_ = np.array(Benchmark_.iloc[2:]), np.array(Rto_.iloc[2:])
                    elif Rto_.iloc[0] != Rto_.iloc[1] and Rto_.iloc[1] == Rto_.iloc[2]:
                        Benchmark_, Rto_ = np.array(Benchmark_.iloc[1:]), np.array(Rto_.iloc[1:])

                    Rto_ = np.array([Rto_[i*3] for i in range(len(Rto_)//3)])

                    opti = scipy.optimize.minimize(
                        fun=_error_function, 
                        x0=[0.5, 1],
                        args=(Benchmark_, Rto_)
                        )
                    self.beta.append(opti.x[0])
                    self.mu.append(opti.x[1])
                    


    def predict(self, Benchmark):
        if isinstance(self.beta, list) and isinstance(self.mu, list):
            Rt = self.mu + self.beta*np.array(Benchmark)[-len(self.beta):]
        else:
            Rt = self.mu + self.beta*np.array(Benchmark)
        Rto_pred = [Rt[0], Rt[1], Rt[2]]
        for i in range(3, len(Rt)):
            Rto_pred.append(np.dot(self.weights.list, np.array(Rt[i-2:i+1])))
        return np.array(Rto_pred)




if __name__ == "__main__":
    # chaining is way faster (almost 2 times)

    alternative_asset_data = (
        # Importing the dataset
        pd.read_excel("/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx", sheet_name= "Alternative Asset")
        # Preprocessing
        .filter(["QUARTER", "Private Equity USD Unhedged"])
        .dropna()
        .assign(returns_PE = (lambda x: x['Private Equity USD Unhedged'].pct_change(fill_method=None)))
        .dropna()
        .set_index("QUARTER")
    )

    classic_asset_data = (
        # Importing the dataset
        pd.read_excel("/Users/adamelbernoussi/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx", sheet_name= "Classic Asset")
        # Preprocessing
        .filter(['QUARTER', 'Date', 'US Equity USD Unhedged'])
        .dropna()
        .set_index('Date', drop = False)
        .resample('M')
        .last()
        .dropna()
        .assign(returns_US_equity = (lambda x: x['US Equity USD Unhedged'].pct_change(fill_method=None)))
        .dropna()
        .set_index("QUARTER")
    )

    results = classic_asset_data.copy()
    results = results.merge(alternative_asset_data, how = 'inner', left_index = True, right_index = True).drop(columns = ['US Equity USD Unhedged', 'Private Equity USD Unhedged'])
    results = results[1:]

    getmansky = GetmanskyModel(2)
    getmansky.set_default_weights("sumOfYears")
    getmansky.fit(results['returns_US_equity'].values.reshape(-1, 1), results['returns_PE'].values.reshape(-1,1), window=24)
    results['returns unsmoothed'] = np.nan
    results['returns unsmoothed'] = getmansky.predict(results['returns_US_equity'])
    results = results.set_index('Date')

    for index, line in results.iterrows():
        if index.month in [1, 2, 4, 5, 7, 8, 10, 11]:
            results.loc[index, 'returns_PE'] = None

    print(getmansky.beta, getmansky.mu)
    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    results['returns PE TR'] = (results['returns_PE']+1).cumprod()-1
    results_no_interpolation = results.resample('QE').last() #just to view the trend
    print(results)
    # Restricting the dates
    end_date_forced = '30-06-2023' #just for the visualisation
    results = results[:end_date_forced]
    results_no_interpolation = results_no_interpolation[:end_date_forced]

    start_date = '2006-08-31'
    end_date = '2010-09-30'
    results_sliced = results.loc[start_date:end_date]

    # Plotting
    # define subplot layout
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('Getmansky model interpolation and unsmoothing PE on US equity', fontsize=12)

    results_no_interpolation['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed', ax=axes[0])
    results_no_interpolation['returns PE TR'].plot(label = 'Rt PE', ax=axes[0])

    results['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed', marker = 'o', linestyle = '', ax=axes[1])
    results['returns PE TR'].plot(label = 'Rt PE', marker = 'o', linestyle = '', ax=axes[1])

    results_sliced['returns unsmoothed TR'].plot(label = 'Rt PE unsmoothed', marker = 'o', linestyle = '', ax=axes[2])
    results_sliced['returns PE TR'].plot(label = 'Rt PE', marker = 'o', linestyle = '', ax=axes[2])

    plt.legend()
    #plt.savefig(f'getmansky/output/GetmanskyPres_8_fev/GetmanskyModel_SoY_{2}_PE_US_equity.png')
    plt.show()

    # an important point here, if we plot quarterly : we have 102 data points
    # and thus monthly is about 303 data points


    ########### to clean ###########

    # # import statsmodels.api as sm
    # # data = sm.datasets.macrodata.load_pandas()
    # # rgdpg = data.data['realgdp'].pct_change().dropna()
    # # acov = sm.tsa.acovf(rgdpg, fft = False, nlag = 2)
    # # theta, sigma2  = sm.tsa.stattools.innovations_algo(acov)
    # # print(acov)
    # # print(theta)