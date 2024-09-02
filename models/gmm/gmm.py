import numpy as np
import random

class GMM:
    def __init__(self, k):
        self.k = k          # Number of gaussian models
        self.data = None    # 2D numpy array
        self.params = None  # Numpy array of (weight, mean, variances) for each gaussian model

    def load_data(self, data):
        self.data = data

    def initialise_params(self):
        # Initialised weights, mean and covariance matrix randomly
        pass

    def fit(self):
        # Expectation step
        # Maximization step
        pass

    def getParams(self):
        return self.params

    def getMembership(self):
        # Returns the value of r_ic
        pass

    def getLikelihood(self):
        # Returns overall likelihood
        pass