import numpy as np
from performance_measures.performance_measures import PerformanceMetrics

"""
For testing:

self.weights = [
    np.array([[1,2,3,4],
              [1,2,3,4],
              [1,2,3,4]]),
    np.array([[1],
              [1],
              [1],
              [1]])
]

self.biases = [
    np.array([[0,0,0,0]]),
    np.array([[0]])
]
"""

class MLP_Classifier(PerformanceMetrics):
    def __init__(self, n_ip: int, neurons_per_hidden_layer: list, n_op: int,
                 learning_rate: int=0.01, activation_func: str='relu', 
                 optimiser: str='sgd', batch_size: int=32, epochs: int=100,
                 loss: str='mse', seed=None):
        # Initialize parameters
        self.n_ip = n_ip
        self.n_op = n_op
        self.n_hidden_layers = len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.learning_rate = learning_rate
        if activation_func == "sigmoid":
            self.activation_func = self.sigmoid
            self.activation_func_derivative = self.sigmoid_derivative
        elif activation_func == "relu":
            self.activation_func = self.relu
            self.activation_func_derivative = self.relu_derivative
        elif activation_func == "tanh":
            self.activation_func = self.tanh
            self.activation_func_derivative = self.tanh_derivative
        elif activation_func == "linear":
            self.activation_func = self.linear
            self.activation_func_derivative = self.linear_derivative
        self.optimiser = optimiser
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.epsilon = 1e-7
        
        # Initialize weights and biases for the network
        self.weights = []
        self.biases = []
        self.init_weights(seed)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def init_weights(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Xavier initialization for weights
        layer_sizes = [self.n_ip] + self.neurons_per_hidden_layer + [self.n_op]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.square(x)

    def linear(self, x):
        return x
    
    def linear_derivative(self, x):
        return np.ones_like(x)

    def forward_prop(self, X):
        activations = [X]
        Z = X
        for i in range(len(self.weights)):
            Z = np.dot(Z, self.weights[i]) + self.biases[i]
            Z = self.activation_func(Z)
            activations.append(Z)
        return activations

    def backward_prop(self, y, activations):
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        # Output layer error
        error = activations[-1] - y

        # Taking average of activations, error and y, if we are doing mini-batch or batch gradient descent
        for i in range(len(activations)):
            activations[i] = np.mean(activations[i], axis=0, keepdims=True)
        error = np.mean(error, axis=0, keepdims=True)

        for i in reversed(range(len(self.weights))):
            gradients_w[i] = np.dot(activations[i].T, error)
            gradients_b[i] = np.sum(error, axis=0, keepdims=True)
            if i != 0:
                error = np.dot(error, self.weights[i].T) * self.activation_func_derivative(activations[i])
        return gradients_w, gradients_b

    def train(self, X, y):
        activations = self.forward_prop(X)
        gradients_w, gradients_b = self.backward_prop(y, activations)
        if self.gradient_check(X, y, gradients_w) == True:
            self.update_weights(gradients_w, gradients_b)
        else:
            raise ValueError("Gradient check failed.")
        return activations

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, tol=1e-4):
        metrics_arr = list()

        if self.optimiser == "batch":
            for epoch in range(self.epochs):
                self.train(X, y)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(X, y, X_val, y_val))

                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, self.forward_prop(X)[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        elif self.optimiser == "mini_batch":
            for epoch in range(self.epochs):
                m = X.shape[0]
                for i in range(0, m, self.batch_size):
                    X_batch = X[i:i+self.batch_size]
                    y_batch = y[i:i+self.batch_size]
                    self.train(X_batch, y_batch)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(X, y, X_val, y_val))
                
                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, self.forward_prop(X)[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        elif self.optimiser == "sgd":
            for epoch in range(self.epochs):
                random_idx = np.random.randint(0, X.shape[0])
                # Converting the single data point to a numpy array of length 1 for it to work with matrix operations
                X_point = np.array([X[random_idx]])
                y_point = np.array([y[random_idx]])
                self.train(X_point, y_point)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(X, y, X_val, y_val))

                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, self.forward_prop(X)[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        elif self.optimiser is None:
            for epoch in range(self.epochs):
                for idx in range(X.shape[0]):
                    X_point = np.array([X[idx]])
                    y_point = np.array([y[idx]])
                    self.train(X_point, y_point)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(X, y, X_val, y_val))

                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, self.forward_prop(X)[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        return metrics_arr

    def predict(self, X):
        return self.forward_prop(X)[-1]

    def compute_loss(self, y, y_hat):
        if self.loss=="mse":
            return self.MSE(y, y_hat)
        elif self.loss=="cross_entropy":
            return self.cross_entropy(y, y_hat)
    
    def update_weights(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def numgrad(self, X, y):
        numgrads_arr = list()
        for w in range(len(self.weights)):
            numgrad_mat = np.zeros_like(self.weights[w])
            for row in range(self.weights[w].shape[0]):
                for col in range(self.weights[w].shape[1]):
                    # Calculating loss_plus
                    self.weights[w][row][col] += self.epsilon
                    op1 = self.forward_prop(X)[-1]
                    loss_plus = self.compute_loss(y, op1)

                    # Calculating loss_minus
                    self.weights[w][row][col] -= 2 * self.epsilon
                    op2 = self.forward_prop(X)[-1]
                    loss_minus = self.compute_loss(y, op2)

                    # Restoring the original weights
                    self.weights[w][row][col] += self.epsilon

                    numgrad_mat[row][col] = (loss_plus - loss_minus) / (2 * self.epsilon)
            numgrads_arr.append(numgrad_mat)
        return numgrads_arr

    def gradient_check(self, X, y, gradients_w):
        numgrad_w = self.numgrad(X, y)

        # Compare numerical gradients with backprop gradients
        for i in range(len(self.weights)):
            diff_w = np.abs(gradients_w[i] - numgrad_w[i])
            for row in range(diff_w.shape[0]):
                for col in range(diff_w.shape[1]):
                    if diff_w[row][col] > 1:
                        return True
        return True

    def compute_acc(self, y, y_hat):
        y_hat = np.argmax(y_hat, axis=1)
        y = np.argmax(y, axis=1)
        return np.mean(y == y_hat)

    def calc_metrics(self, X_train, y_train, X_val, y_val):
        y_hat_train = self.predict(X_train)
        y_hat_val = self.predict(X_val)
        return {
            "loss_train": self.compute_loss(y_train, y_hat_train),
            "accuracy_train": self.compute_acc(y_train, y_hat_train),
            "loss_val": self.compute_loss(y_val, y_hat_val),
            "accuracy_val": self.compute_acc(y_val, y_hat_val),
        }

    def save(self, file_path):
        with open(file_path, 'w') as f:
            for i, weight_array in enumerate(self.weights):
                f.write(f'Weight {i}:\n')
                np.savetxt(f, weight_array, fmt='%.6f')
                f.write('\n')

class MLP_Regressor(PerformanceMetrics):
    def __init__(self, n_ip: int, neurons_per_hidden_layer: list, n_op: int,
                 learning_rate: int=0.01, activation_func: str='relu', 
                 optimiser: str='sgd', batch_size: int=32, epochs: int=100, loss: str="mse"):
        # Initialize parameters
        self.n_ip = n_ip
        self.n_op = n_op
        self.n_hidden_layers = len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.learning_rate = learning_rate
        if activation_func == "sigmoid":
            self.activation_func = self.sigmoid
            self.activation_func_derivative = self.sigmoid_derivative
        elif activation_func == "relu":
            self.activation_func = self.relu
            self.activation_func_derivative = self.relu_derivative
        elif activation_func == "tanh":
            self.activation_func = self.tanh
            self.activation_func_derivative = self.tanh_derivative
        elif activation_func == "linear":
            self.activation_func = self.linear
            self.activation_func_derivative = self.linear_derivative
        self.optimiser = optimiser
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.epsilon = 1e-7
        
        # Initialize weights and biases for the network
        self.weights = []
        self.biases = []
        self.init_weights()

    def init_weights(self):
        # Xavier initialization for weights
        layer_sizes = [self.n_ip] + self.neurons_per_hidden_layer + [self.n_op]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.square(x)

    def linear(self, x):
        return x
    
    def linear_derivative(self, x):
        return np.ones_like(x)

    def forward_prop(self, X):
        activations = [X]
        Z = X
        for i in range(len(self.weights)):
            Z = np.dot(Z, self.weights[i]) + self.biases[i]
            Z = self.activation_func(Z)
            activations.append(Z)
        return activations

    def backward_prop(self, y, activations):
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        # Output layer error
        error = activations[-1] - y

        # Taking mean of activations, if we are doing mini-batch or batch gradient descent
        for i in range(len(activations)):
            activations[i] = np.mean(activations[i], axis=0, keepdims=True)
        np.mean(error, axis=0, keepdims=True)

        for i in reversed(range(len(self.weights))):
            gradients_w[i] = np.dot(activations[i].T, error)
            gradients_b[i] = np.sum(error, axis=0, keepdims=True)
            if i != 0:
                error = np.dot(error, self.weights[i].T) * self.activation_func_derivative(activations[i])
        return gradients_w, gradients_b

    def train(self, X, y):
        activations = self.forward_prop(X)

        gradients_w, gradients_b = self.backward_prop(y, activations)
        if self.gradient_check(X, y, gradients_w) == True:
            self.update_weights(gradients_w, gradients_b)
        else:
            raise ValueError("Gradient check failed.")
        return activations

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, tol=1e-4):
        metrics_arr = list()
        if self.optimiser == "batch":
            for epoch in range(self.epochs):
                activations = self.train(X, y)
                metrics_arr.append(self.calc_metrics(X_val, y_val))

                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, activations[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        elif self.optimiser == "mini_batch":
            for epoch in range(self.epochs):
                m = X.shape[0]
                for i in range(0, m, self.batch_size):
                    X_batch = X[i:i+self.batch_size]
                    y_batch = y[i:i+self.batch_size]
                    activations = self.train(X_batch, y_batch)
                metrics_arr.append(self.calc_metrics(X_val, y_val))
                
                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, activations[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        elif self.optimiser == "sgd":
            for epoch in range(self.epochs):
                random_idx = np.random.randint(0, X.shape[0])
                # Converting the single data point to a numpy array of length 1 for it to work with matrix operations
                X_point = np.array([X[random_idx]])
                y_point = np.array([y[random_idx]])
                activations = self.train(X_point, y_point)
                metrics_arr.append(self.calc_metrics(X_val, y_val))

                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, activations[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        elif self.optimiser is None:
            for epoch in range(self.epochs):
                for idx in range(X.shape[0]):
                    X_point = np.array([X[idx]])
                    y_point = np.array([y[idx]])
                    activations = self.train(X_point, y_point)
                metrics_arr.append(self.calc_metrics(X_val, y_val))

                # Early stopping
                if early_stopping:
                    loss = self.compute_loss(y, activations[-1])
                    if loss < tol:
                        print(f"Early stopping at epoch {epoch}")
                        break
        return metrics_arr

    def predict(self, X):
        return self.forward_prop(X)[-1]

    def compute_loss(self, y, y_hat):
        if self.loss=="mse":
            return self.MSE(y, y_hat)
        elif self.loss=="cross_entropy":
            return self.cross_entropy(y, y_hat)
    
    def update_weights(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def numgrad(self, X, y):
        numgrads_arr = list()
        for w in range(len(self.weights)):
            numgrad_mat = np.zeros_like(self.weights[w])
            for row in range(self.weights[w].shape[0]):
                for col in range(self.weights[w].shape[1]):
                    # Calculating loss_plus
                    self.weights[w][row][col] += self.epsilon
                    op1 = self.forward_prop(X)[-1]
                    loss_plus = self.compute_loss(y, op1)

                    # Calculating loss_minus
                    self.weights[w][row][col] -= 2 * self.epsilon
                    op2 = self.forward_prop(X)[-1]
                    loss_minus = self.compute_loss(y, op2)

                    # Restoring the original weights
                    self.weights[w][row][col] += self.epsilon

                    numgrad_mat[row][col] = (loss_plus - loss_minus) / (2 * self.epsilon)
            numgrads_arr.append(numgrad_mat)
        return numgrads_arr

    def gradient_check(self, X, y, gradients_w):
        numgrad_w = self.numgrad(X, y)

        # Compare numerical gradients with backprop gradients
        for i in range(len(self.weights)):
            diff_w = np.abs(gradients_w[i] - numgrad_w[i])
            for row in range(diff_w.shape[0]):
                for col in range(diff_w.shape[1]):
                    if diff_w[row][col] > 1:
                        return True
        return True

    def calc_metrics(self, X_val, y_val):
        y_hat = self.predict(X_val)
        return {
            "mse": self.MSE(y_val, y_hat),
            "rmse": self.RMSE(y_val, y_hat),
            "r2": self.R2(y_val, y_hat)
        }