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

# Importing the dataset
alternative_asset_data = pd.read_excel('/Users/adam/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Alternative Asset')
classic_asset_data = pd.read_excel('/Users/adam/Desktop/EnsaeAlternativeSubject/EnsaeAlternativeTimeSeries.xlsx', sheet_name= 'Classic Asset')


