"""
In this file we will implement all the functions necessary to compute different types of weights.
And also every functions that computes metrics from weights.
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We will now create some functions that will return a list of weights

class Weights:
    """In this class we will code every functions related to the weights of the Getmansky model.
    The weights can be of three types:
    - Equal weights
    - Sum of years weights
    - Geometric weights
    """

    def __init__(
        self, 
        type, 
        k, 
        delta=None
    ):
        """
        Initializes the Weights object.
        Parameters:
        -----------
        type: str
            The type of weights to be used. It can be "equal", "sumOfYears" or "geometric".
        k: int
            The number of weights to be used.
        delta: float
            The parameter to be used in the geometric weights.
        """
        if type == "equal":
            self.list = [1/(k+1) for _ in range(k+1)]
        elif type == "sumOfYears":
            self.list = [(k+1-i)/((k+1)*(k+2)/2) for i in range(k+1)]
        elif type == "geometric":
            self.list = [((delta**i)*(1-delta))/(1-delta**(k+1)) for i in range(k+1)]
        else:   
            raise ValueError("The type of weights is not valid")
        self.type = type
        self.k = k
        self.delta = delta

    def __str__(self):
        """
        This function is here to facilitate the code reading
        """
        return str(self.list)
    
    def c_mu(self):
        """
        This function computes the sum of the weights
        """
        return np.sum(self.list)
    
    def c_sigma(self):
        return np.sqrt(np.sum(np.array(self.list)**2))
    
    def c_s(self):
        return 1/self.c_sigma()
    
    def xi(self):
        """
        This function is here to facilitate the code reading
        It can be seen as a "smoothing index"
        """
        return np.sum(np.array(self.list)**2)

    def zeta(self):
        return np.sum(np.array([1-np.sum(self.list[:i]) for i in range(self.k)])**2)



if __name__ == "__main__":
    weights = Weights("equal", 5)
    print(weights)
    weights = Weights("sumOfYears", 2)
    print(weights)
    weights = Weights("geometric", 5, 0.5)
    print(weights)
    print(weights.c_mu())
    print(weights.c_sigma())
    print(weights.c_s())
    print(weights.xi())
    print(weights.zeta())