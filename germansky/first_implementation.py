import numpy as np
import scipy

#Time-varying expected returns
#Markov process:

class two_state_markov_chain:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.kernel = np.array([[p,1-p], [q, 1-q]])
        self.expected_value = (1-q)/(2-p-q)
        self.var = ((1-p)*(1-q))/(2-p-q)**2

    def get_invariant_law(self):
        return np.array([(1-self.p)/(2-self.p-self.q), (1-self.q)/(2-self.p-self.q)])
    

print(two_state_markov_chain(0.3, 0.4).get_invariant_law())


