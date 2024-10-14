import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.MLP.MLP import MLP

class AutoEncoder(MLP):
    def __init__(self, n_ip: int=0, neurons_per_hidden_layer: list=[], n_op: int=0,
                 learning_rate: int=0.01, activation_func: str='relu', 
                 optimiser: str='sgd', batch_size: int=32, epochs: int=100,
                 loss: str='mse', seed=None):
        super().__init__(n_ip, neurons_per_hidden_layer, n_op, learning_rate, activation_func, optimiser, batch_size, epochs, loss, seed)

    def fit(self, X):
        super().fit(X, X)

    def get_latent(self, X):
        activations = super().forward_prop(X)
        latent = activations[len(activations) // 2]
        return latent

