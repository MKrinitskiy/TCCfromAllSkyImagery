import numpy as np


class binomial_proba_calculator:
    def __init__(self, classes_number = 9):
        self.K = classes_number
        self.k = np.arange(1, self.K+1,1)
        self.mult_factors = np.array([np.math.factorial(self.K-1)/np.math.factorial(k-1)/np.math.factorial(self.K-k) for k in range(1, self.K+1)])

    def __call__(self, p):
        return self.mult_factors*np.power(p, self.k-1)*np.power((1-p), (self.K-self.k))