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
alpha = 1
phi = -0.45495298342879176
gamma = 0.020106161007845565 
for i in range(2, 50):
    Rto_pred.append(gamma*(1-alpha) + (alpha+phi)*Rto_pred[i-1] - alpha*phi*Rto_pred[i-2])


print(Rto_pred)
print(gamma*(1-alpha) + (alpha+phi)*20 - alpha*phi*20)
plt.plot(list(range(len(Rto_pred))), Rto_pred)
plt.show()