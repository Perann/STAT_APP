import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys



Rto_pred = [0.5, 1]
alpha = 0.34705685478630866
phi = -0.23199383553719802
gamma = 0.01411024243469647
for i in range(2, 50):
    Rto_pred.append(gamma*(1-alpha) + (alpha+phi)*Rto_pred[-1])# - alpha*phi*Rto_pred[i-2])
    print(Rto_pred)


Rto_pred = np.array(Rto_pred)
Rto_shift = Rto_pred[1:]
Rto_pred = Rto_pred[:-1]
Rt = (Rto_shift - alpha*Rto_pred)/(1-alpha)
plt.plot(list(range(len(Rt))), Rt)
plt.show()