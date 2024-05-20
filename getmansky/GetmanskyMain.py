"""
In this file we will implement a first version of the getmansky model.
We will implement the functions to optimize the weights and to fit the model.
The structure of the code is quit similar to what you see in sk-learn code.
There is a class GetmanskyModel that will contain all the functions.
There is a function fit that will fit the model. And a function predict that will predict the returns.
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from types import NoneType
import warnings

# Importing packages from the project (we put a try to handle path error)
try:
    from ..getmansky.WeightsFunctions.weights import Weights
except ImportError:
    from WeightsFunctions.weights import Weights

# Class of the Getmansky model
class GetmanskyModel:
    """
    This class will contain all the functions related to the Getmansky model.
    
    The Getmansky model is a model that predicts the returns of an alternative asset using the returns of a benchmark and the observed return of the alternative asset (less frequently).

    Parameters:
    -----------
    k: int
        The memory of the model. It is the number of weights to be used minus 1.

    Examples:
    ---------
    >>> getmansky = GetmanskyModel(2)
    >>> getmansky.set_default_weights("sumOfYears")
    >>> getmansky.fit(df["return_benchmark"].values.reshape(-1, 1), df["return_alternative_asset"].values.reshape(-1,1), window = None)
    >>> df["return_unsmoothed"] = getmansky.predict(df["return_benchmark"], rebase = df["returns_alternative_asset"])
    """
    def __init__(
        self, 
        k : int = 2
    ) -> None:
        """
        Initializes the GetmanskyModel object.

        Parameters:
        -----------
        k: int
            The memory of the model. It is the number of weights to be used minus 1.
        """
        self.k = k
        self.weights = Weights("equal", k) # default value
        self.mu = 0
        self.beta = 1

    def set_default_weights(self, type_, delta=None) -> None:
        """
        This function sets the default weights of the model.
        In order to use the getmanky model you should use either this function or the optimize_weights function.

        Parameters:
        -----------
        type_: str
            The type of weights to be used. It can be "equal", "sumOfYears" or "geometric".
        delta: float | None
            The parameter to be used in the geometric weights.
        """
        # updating weight list
        self.weights = Weights(type_, self.k, delta)

    def optimize_weights_MLE(self, Rto):
        """
        Didn't have time to implement this function.
        """
        pass

    def optimize_weights_LR(self, Benchmark, Rto) -> None:
        """
        This function optimizes the weights of the model using a linear regression.

        Parameters:
        -----------
        Benchmark: np.array
            The benchmark returns.
        Rto: np.array
            The observed returns of the alternative asset.
        """
        Benchmark_, Rto_ = Benchmark, Rto
        # This is to determine if we are at the beginning of the semester and hence rebase or not
        if Rto_[0] == Rto_[1] and Rto_[1] == Rto_[2]:
            Benchmark_, Rto_ = np.array(Benchmark_), np.array(Rto_)                        
        elif Rto_[0] == Rto_[1] and Rto_[1] != Rto_[2]:
            Benchmark_, Rto_ = np.array(Benchmark_[2:]), np.array(Rto_[2:])
        elif Rto_[0] != Rto_[1] and Rto_[1] == Rto_[2]:
            Benchmark_, Rto_ = np.array(Benchmark_[1:]), np.array(Rto_[1:])
        rto, benchmark = pd.Series(Rto_), pd.Series(Benchmark_)
        df = pd.DataFrame([benchmark, rto], index = ['Benchmark', 'Rto']).T
        for i in range(1, self.k+1):
            df[f'bench_lag_{i}'] = df['Benchmark'].shift(i)
        df["1"] = 1 # adding the intercept column
        df.dropna(inplace = True)
        X, y = df.drop(columns = ['Rto']), df['Rto']
    
        def _error_function(x, X, y):
            X = np.dot(X, x)
            return np.sum((y[2::3] - np.array([(1+X[3*i])*(1+X[3*i+1])*(1+X[3*i+2])-1 for i in range(len(X)//3)]))**2)
        
        opti = scipy.optimize.minimize(
                fun=_error_function, 
                x0=[0.5]*(self.k + 2),
                args = (X, y),
                bounds = ((0,None),(0,None), (0,None), (None, None))
            )
        if not opti.success: warnings.warn(f"The optimisation of {Benchmark} didn't terminated successfuly !")
        # updating weight list
        self.weights.list = opti.x[:(self.k+1)]/np.sum(opti.x[:(self.k+1)])

    def fit(self, Benchmark, Rto, window : int =None) -> None:
        """
        This function fits the model to the data.
        More precisely, it optimizes the mu and beta parameters of the model.
        If there is a not None window argument, the optimization is done on a rolling window. And the beta and mu are lists.
        If not, the optimization is done on the whole dataset. And the beta and mu are scalars.

        Parameters:
        -----------
        Benchmark: np.array
            The benchmark returns you want as a reference.
        Rto: np.array
            The observed returns of the alternative asset.
        window: int | None
            The window to use for the rolling optimization. If None, the optimization is done on the whole dataset.
        """
        # Checking type
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
            
            if not opti.success: warnings.warn(f"The optimisation of {Benchmark} didn't terminated successfuly !")

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
                    
                    if not opti.success: warnings.warn(f"The optimisation of {Benchmark} didn't terminated successfuly !")

                    self.beta.append(opti.x[0])
                    self.mu.append(opti.x[1])

    def predict(self, Benchmark, rebase = None) -> np.array:
        """
        This function predicts the returns of the alternative asset.

        Parameters:
        -----------
        Benchmark: np.array
            The benchmark returns you want as a reference.
        rebase: np.array | None
            The rebase parameter. If None, no rebase is done. If not None, the rebase is done on the passed array.

        Returns:
        --------
        Rt : np.array
            The predicted returns of the alternative asset.
        """
        if isinstance(self.beta, list) and isinstance(self.mu, list):
            Rt = self.mu + self.beta*np.array(Benchmark)[-len(self.beta):]
            rebase = rebase[-len(self.beta):]
        else:
            Rt = self.mu + self.beta*np.array(Benchmark)
        if isinstance(rebase, np.ndarray):
            # handling edge cases of first values
            if rebase[0] == rebase[1] and rebase[1] == rebase[2]:
                start = 2 # basicly nothing to do around here                      
            elif rebase[0] == rebase[1] and rebase[1] != rebase[2]:
                start = 4
                Rt[1] = ((1+rebase[1])/(1+Rt[0])) - 1
            elif rebase[0] != rebase[1] and rebase[1] == rebase[2]:
                start = 3
                Rt[0] = rebase[0]
            for i in range(start, len(rebase), 3):
                Rt[i] = ((1+rebase[i])/((1+Rt[i-2])*(1+Rt[i-1]))) - 1
                
        elif rebase is not None:
            raise ValueError("Warning ! rebase argument should be None (no rebase) or a np.ndarray")
        
        return np.array(Rt)




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
        .resample('ME')
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
    getmansky.fit(results['returns_US_equity'].values.reshape(-1, 1), results['returns_PE'].values.reshape(-1,1))
    results['returns unsmoothed'] = getmansky.predict(results['returns_US_equity'])

    results = results.set_index('Date')

    for line in results.iterrows():
        if line[0].month in [1, 2, 4, 5, 7, 8, 10, 11]:
            line[1]['returns_PE'] = None

    results['returns unsmoothed TR'] = (results['returns unsmoothed']+1).cumprod()-1
    results['returns PE TR'] = (results['returns_PE']+1).cumprod()-1
    results_no_interpolation = results.resample('QE').last() #just to view the trend

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