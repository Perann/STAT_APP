"""
In this file we will implement all the functions necessary to compute different types of weights.
And also every functions that computes metrics from weights.
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We will now create some functions that will return a list of weights
def equal(k : int):
    """
    This function will return a list of weights of size k+1 (most recent k+1 periods, including the current one)
    """
    return [1/(k+1) for _ in range(k+1)]

def SumOfYears(k : int):
    """
    This function will return a list of weights of size k+1 (most recent k+1 periods, including the current one)
    """
    return [(k+1-i)/((k+1)*(k+2)/2) for i in range(k+1)]

def Geometric(k : int, delta : float):
    """
    This function will return a list of weights of size k+1 (most recent k+1 periods, including the current one)
    """
    return [((delta**i)*(1-delta))/(1-delta**(k+1)) for i in range(k+1)]


if __name__ == "__main__":
    print(equal(5))
    print(SumOfYears(5))
    print(Geometric(5, 0.5))