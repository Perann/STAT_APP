import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from autoreg_coeff import auto_reg

def get_alpha(datas):
    gamma, phi = auto_reg(datas)
    def function_to_minimize(alpha):
        res = 0
        for t in range(2,len(datas)):
            res += (datas[t] - gamma*(1-alpha) -(alpha + phi)*datas[t-1] - alpha*phi*datas[t-2])**2
        return res
    return scipy.optimize.minimize(function_to_minimize,1/2).x[0]


    