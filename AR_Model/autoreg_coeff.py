import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def auto_reg(serie,p_order = 1, intergrate = 0, moving_average = 0):
    model = ARIMA(serie, order=(1, 0, 0))
    results = model.fit()
    coeff = results.params
    return (coeff[0], coeff[1])

