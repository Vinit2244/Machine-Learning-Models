import numpy as np

class GMM:
    def __init__(self, k):
        self.k = k
        self.data = None
        pass

    def load_data(self, data):
        self.data = data
