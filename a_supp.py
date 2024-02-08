import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys



gamma = 0.014053217162944536
phi = -0.3169362753345105
alpha = 0.470059725508072

rt = [0.01, 0.088]
for _ in range(2, 50):
    rt.append(gamma+phi*rt[-1])

rt = np.array(rt)

Rto = [0.01, 0.088]
for i in range(2, 50):
    Rto.append(alpha*Rto[-1]+(1-alpha)*rt[i])



plt.plot(list(range(len(rt))), rt)
plt.show()