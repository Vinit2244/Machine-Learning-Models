import numpy as np
from performance_measures.performance_measures import PerformanceMetrics

class MLP(PerformanceMetrics):
    def __init__(self, n_ip: int=0, neurons_per_hidden_layer: list=[], n_op: int=0,
                 learning_rate: int=0.01, activation_func: str='relu', 
                 optimiser: str='sgd', batch_size: int=32, epochs: int=100,
                 loss: str='mse', seed=None, logistic_reg=False):
        # Initialize parameters
        self.n_ip = n_ip
        self.n_op = n_op
        self.n_hidden_layers = len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.learning_rate = learning_rate
        self.af = activation_func
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
        self.logistic_reg = logistic_reg
        
        # Initialize weights and biases for the network
        self.weights = []
        self.biases = []
        if n_ip != 0 and n_op != 0:
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
        final_gradients_w = [None] * len(self.weights)
        final_gradients_b = [None] * len(self.biases)

        # Output layer error for all data samples
        error = activations[-1] - y

        # Backpropagating the error for each individual data sample
        for data_sample_idx in range(y.shape[0]):
            err = error[data_sample_idx].reshape(1, -1)  # Error for the current data sample
            for i in reversed(range(len(self.weights))): # Looping through the layers in reverse order
                activation = activations[i][data_sample_idx].reshape(1, -1)
                gradients_wi = np.dot(activation.T, err)
                gradients_bi = np.sum(err, axis=0, keepdims=True)
                if i != 0:
                    err = np.dot(err, self.weights[i].T) * self.activation_func_derivative(activation)
                final_gradients_w[i] = gradients_wi if final_gradients_w[i] is None else final_gradients_w[i] + gradients_wi
                final_gradients_b[i] = gradients_bi if final_gradients_b[i] is None else final_gradients_b[i] + gradients_bi
        # Taking average of gradients for all data samples
        final_gradients_w = [grad / y.shape[0] for grad in final_gradients_w]
        final_gradients_b = [grad / y.shape[0] for grad in final_gradients_b]

        # Taking average of activations, error and y, if we are doing mini-batch or batch gradient descent
        # for i in range(len(activations)):
        #     activations[i] = np.mean(activations[i], axis=0, keepdims=True)
        # error = np.mean(error, axis=0, keepdims=True)

        # for i in reversed(range(len(self.weights))):
        #     gradients_w[i] = np.dot(activations[i].T, error)
        #     gradients_b[i] = np.sum(error, axis=0, keepdims=True)
        #     if i != 0:
        #         error = np.dot(error, self.weights[i].T) * self.activation_func_derivative(activations[i])
        
        return final_gradients_w, final_gradients_b

    def train(self, X, y):
        activations = self.forward_prop(X)
        gradients_w, gradients_b = self.backward_prop(y, activations)
        self.update_weights(gradients_w, gradients_b)
        # if self.gradient_check(X, y, gradients_w) == True:
        #     self.update_weights(gradients_w, gradients_b)
        # else:
        #     raise ValueError("Gradient check failed.")
        return activations

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, multi_class=False):
        metrics_arr = list()
        best_weights = None
        val_errors = list()
        min_val_loss = float('inf')

        if self.optimiser == "batch":
            for epoch in range(self.epochs):
                self.train(X, y)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(epoch, X, y, X_val, y_val, multi_class))

                # Early stopping
                if early_stopping:
                    val_loss = self.compute_loss(y_val, self.forward_prop(X_val)[-1])
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_weights = self.weights.copy()
                    if len(val_errors) < 5:
                        val_errors.append(val_loss)
                    else:
                        val_errors.pop(0)
                        val_errors.append(self.compute_loss(y_val, self.forward_prop(X_val)[-1]))
                        if val_errors[0] < val_errors[1] < val_errors[2] < val_errors[3] < val_errors[4]:
                            print(f"Early stopping at epoch {epoch}")
                            self.weights = best_weights
                            break
        
        elif self.optimiser == "mini_batch":
            for epoch in range(self.epochs):
                m = X.shape[0]
                for i in range(0, m, self.batch_size):
                    X_batch = X[i:i+self.batch_size]
                    y_batch = y[i:i+self.batch_size]
                    self.train(X_batch, y_batch)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(epoch, X, y, X_val, y_val, multi_class))
            
                
                # Early stopping
                if early_stopping:
                    val_loss = self.compute_loss(y_val, self.forward_prop(X_val)[-1])
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_weights = self.weights.copy()
                    if len(val_errors) < 5:
                        val_errors.append(val_loss)
                    else:
                        val_errors.pop(0)
                        val_errors.append(self.compute_loss(y_val, self.forward_prop(X_val)[-1]))
                        if val_errors[0] < val_errors[1] < val_errors[2] < val_errors[3] < val_errors[4]:
                            print(f"Early stopping at epoch {epoch}")
                            self.weights = best_weights
                            break
        
        elif self.optimiser == "sgd":
            for epoch in range(self.epochs):
                random_idx = np.random.randint(0, X.shape[0])
                # Converting the single data point to a numpy array of length 1 for it to work with matrix operations
                X_point = np.array([X[random_idx]])
                y_point = np.array([y[random_idx]])
                self.train(X_point, y_point)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(epoch, X, y, X_val, y_val, multi_class))

                # Early stopping
                if early_stopping:
                    val_loss = self.compute_loss(y_val, self.forward_prop(X_val)[-1])
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_weights = self.weights.copy()
                    if len(val_errors) < 5:
                        val_errors.append(val_loss)
                    else:
                        val_errors.pop(0)
                        val_errors.append(self.compute_loss(y_val, self.forward_prop(X_val)[-1]))
                        if val_errors[0] < val_errors[1] < val_errors[2] < val_errors[3] < val_errors[4]:
                            print(f"Early stopping at epoch {epoch}")
                            self.weights = best_weights
                            break
        
        elif self.optimiser is None:
            for epoch in range(self.epochs):
                for idx in range(X.shape[0]):
                    X_point = np.array([X[idx]])
                    y_point = np.array([y[idx]])
                    self.train(X_point, y_point)
                if X_val is not None and y_val is not None:
                    metrics_arr.append(self.calc_metrics(epoch, X, y, X_val, y_val, multi_class))

                # Early stopping
                if early_stopping:
                    val_loss = self.compute_loss(y_val, self.forward_prop(X_val)[-1])
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_weights = self.weights.copy()
                    if len(val_errors) < 5:
                        val_errors.append(val_loss)
                    else:
                        val_errors.pop(0)
                        val_errors.append(self.compute_loss(y_val, self.forward_prop(X_val)[-1]))
                        if val_errors[0] < val_errors[1] < val_errors[2] < val_errors[3] < val_errors[4]:
                            print(f"Early stopping at epoch {epoch}")
                            self.weights = best_weights
                            break
        return metrics_arr

    def predict(self, X, multi_class=False):
        # If regression model
        if self.n_op == 1:
            return self.forward_prop(X)[-1]
        elif multi_class:
            return self.forward_prop(X)[-1]
        # If classification model
        else:
            return self.softmax(self.forward_prop(X)[-1])

    def compute_loss(self, y, y_hat):
        if self.loss=="mse":
            return self.MSE(y, y_hat)
        elif self.loss=="cross_entropy":
            return self.cross_entropy(y, y_hat)
        elif self.loss=="hamming_loss":
            y_hat = np.where(y_hat > 0.5, 1, 0)
            return self.hamming_loss(y, y_hat)
    
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
                        return False
        return True

    def compute_acc(self, y, y_hat, multi_class=False):
        if multi_class:
            y_hat = np.where(y_hat > 0.5, 1, 0)
            n_classes = y.shape[1]
            n_correct = 0
            n_total = 0
            for cls in range(n_classes):
                y_cls = y[:, cls]
                y_hat_cls = y_hat[:, cls]
                for i in range(len(y_cls)):
                    if y_cls[i] == 1:
                        n_total += 1
                        if y_hat_cls[i] == 1:
                            n_correct += 1
            return n_correct / n_total
        else:
            y_hat = np.argmax(y_hat, axis=1)
            y = np.argmax(y, axis=1)
            return np.mean(y == y_hat)

    def calc_metrics(self, epoch, X_train, y_train, X_val, y_val, multi_class=False):
        y_hat_train = self.predict(X_train, multi_class)
        y_hat_val = self.predict(X_val, multi_class)
        if self.logistic_reg:
            y_hat_train = self.sigmoid(y_hat_train)
            y_hat_val = self.sigmoid(y_hat_val)
            return {
                "mse_train": self.MSE(y_train, y_hat_train),
                "cross_entropy_train": self.cross_entropy(y_train, y_hat_train),
                "mse_val": self.MSE(y_val, y_hat_val),
                "cross_entropy_val": self.cross_entropy(y_val, y_hat_val),
                "epoch": epoch
            }
        elif self.n_op == 1:
            return {
                "mse_train": self.MSE(y_train, y_hat_train),
                "mse_val": self.MSE(y_val, y_hat_val),
                "epoch": epoch
            }
        else:
            return {
                "loss_train": self.compute_loss(y_train, y_hat_train),
                "accuracy_train": self.compute_acc(y_train, y_hat_train, multi_class),
                "loss_val": self.compute_loss(y_val, y_hat_val),
                "accuracy_val": self.compute_acc(y_val, y_hat_val, multi_class),
                "epoch": epoch
            }

    def save_model(self, file_path):
        with open(file_path, 'w') as f:
            # Save architecture-related attributes
            f.write(f"input: {self.n_ip}\n")
            f.write(f"hidden: {self.neurons_per_hidden_layer}\n")
            f.write(f"output: {self.n_op}\n")

            # Save hyperparameters and other model attributes
            f.write(f"learning_rate: {self.learning_rate}\n")
            f.write(f"activation_function: {self.af}\n")
            f.write(f"optimiser: {self.optimiser}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"epochs: {self.epochs}\n")
            f.write(f"loss: {self.loss}\n")
            f.write(f"logistic_reg: {self.logistic_reg}\n")

            # Save weights
            for i, weight_array in enumerate(self.weights):
                f.write(f'Weight {i}:\n')
                np.savetxt(f, weight_array, fmt='%.6f')
                f.write('\n')

            # Save biases
            for i, bias_array in enumerate(self.biases):
                f.write(f'Bias {i}:\n')
                np.savetxt(f, bias_array, fmt='%.6f')
                f.write('\n')

    def load_model(self, file_path):
        self.weights = []
        self.biases = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Load architecture-related attributes
            self.n_ip = int(lines[0].split(":")[1].strip())
            self.neurons_per_hidden_layer = eval(lines[1].split(":")[1].strip())
            self.n_op = int(lines[2].split(":")[1].strip())

            # Load hyperparameters and other model attributes
            self.learning_rate = float(lines[3].split(":")[1].strip())
            self.af = lines[4].split(":")[1].strip()
            self.optimiser = lines[5].split(":")[1].strip()
            if lines[6].split(":")[1].strip() == "None":
                self.batch_size = None
            else:
                self.batch_size = int(lines[6].split(":")[1].strip())
            self.epochs = int(lines[7].split(":")[1].strip())
            self.loss = lines[8].split(":")[1].strip()
            self.logistic_reg = lines[9].split(":")[1].strip().lower() == 'true'

            current_weight = []
            current_bias = []
            is_weight = True  # Flag to know if we're reading weights or biases

            for line in lines[10:]:
                if line.startswith('Weight'):
                    if current_weight:
                        self.weights.append(np.array(current_weight))
                        current_weight = []
                    is_weight = True
                elif line.startswith('Bias'):
                    if current_bias:
                        self.biases.append(np.array(current_bias))
                        current_bias = []
                    is_weight = False
                else:
                    stripped_line = line.strip()
                    if stripped_line:
                        if is_weight:
                            current_weight.append([float(x) for x in stripped_line.split()])
                        else:
                            current_bias.append([float(x) for x in stripped_line.split()])

            # Append the last set of weights and biases
            if current_weight:
                self.weights.append(np.array(current_weight))
            if current_bias:
                self.biases.append(np.array(current_bias))

