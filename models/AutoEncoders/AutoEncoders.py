import numpy as np

class AutoEncoder:
    def __init__(self, layers: list, learning_rate: float, epochs: int, batch_size: int):
        self.n_ip = layers[0]
        self.n_op = layers[-1]
        self.layers = layers

