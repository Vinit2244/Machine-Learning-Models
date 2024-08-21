import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from performance_measures.performance_measures import PerformanceMetrics

class Regression(PerformanceMetrics):
    def __init__(self, regularization_method=None, lamda=0):
        super().__init__()
        self.params = None # Stored in increasing order of degree
        self.data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.min_err_test_k = None
        self.min_err_test_params = None
        self.min_err = None
        self.lamda = lamda
        self.regularization_method = regularization_method
        self.img_arr = list()
        self.var = None
        self.sd = None
    
    def load_model(self, path):
        with open(path, "r") as f:
            params = f.read().split(",")
            self.k = len(params) - 1
            self.params = np.array([float(param) for param in params])
    
    def load_data(self, data):
        self.data = data
        self.var = self.variance(data[:,1])
        self.sd = self.standard_deviation(data[:,1])
    
    def shuffle_data(self):
        np.random.shuffle(self.data)
    
    def split_data(self, train, validation, test):
        if train + validation + test != 100:
            raise ValueError("Train, validation and test must sum to 100")
        
        n_total_samples = self.data.shape[0]
        n_train_samples = int((train/100) * n_total_samples)
        n_validation_samples = int((validation/100) * n_total_samples)
        n_test_samples = int((test/100) * n_total_samples)

        self.train_data = self.data[0:n_train_samples]
        self.validation_data = self.data[n_train_samples:n_train_samples+n_validation_samples]
        self.test_data = self.data[n_train_samples+n_validation_samples:]

    def visualise_split(self):
        x_train = self.train_data[:,0]
        y_train = self.train_data[:,1]
        x_validation = self.validation_data[:,0]
        y_validation = self.validation_data[:,1]
        x_test = self.test_data[:,0]
        y_test = self.test_data[:,1]

        # Plotting the scatter plot of train, validation and test data split
        plt.scatter(x_train, y_train, color='red', label='Train', s=10)
        plt.scatter(x_validation, y_validation, color='blue', label='Validation', s=10)
        plt.scatter(x_test, y_test, color='green', label='Test', s=10)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatter plot of train, validation and test split")
        plt.legend()
        plt.show()

    def func(self, x):
        if self.params is None:
            raise ValueError("Model has not been initialised properly yet")
        y_pred = 0
        x_powered = 1
        for i in range(len(self.params)):
            y_pred += self.params[i] * x_powered
            x_powered *= x
        return y_pred
    
    def fit(self, k, lr, threshold, save_epoch_imgs=False, seed=42, max_epochs=float('inf')):
        self.img_arr = list()
        # Note here for the value of y intercept we can also use the y value of the first point for faster convergence
        np.random.seed(seed)
        self.params = np.random.rand(k + 1)
        errors = list()
        epoch = -1
        while len(errors) < 2 or abs(errors[-1][0] - errors[-2][0]) > threshold and epoch < max_epochs:
            epoch += 1
            x = self.train_data[:,0]
            y = self.train_data[:,1]
            y_pred = np.zeros(len(x))
            for i in range(len(x)):
                y_pred[i] = self.func(x[i])
            y_error = y_pred - y

            mse = self.MSE(y, y_pred)
            var = self.variance(y_pred)
            sd = self.standard_deviation(y_pred)
            errors.append([mse, var, sd])

            for i in range(k + 1):
                final_sum = 0
                for j in range(len(y_error)):
                    final_sum += y_error[j] * (x[j] ** i)

                # Adding regularization term
                if self.regularization_method == "l1":
                    # Source: https://medium.com/intuition/understanding-l1-and-l2-regularization-with-analytical-and-probabilistic-views-8386285210fc
                    # Applying soft-thresholding
                    if self.params[i] > 0:
                        final_sum += self.lamda
                    elif self.params[i] < 0:
                        final_sum -= self.lamda

                elif self.regularization_method == "l2":
                    final_sum += 2 * self.lamda * self.params[i]

                self.params[i] = self.params[i] - lr * final_sum / len(y_error)
            if save_epoch_imgs:
                self.visualise_fit("save in arr", k, epoch, "temp.png", errors)
        mse, var, sd = self.get_metrics("test")
        if self.min_err is None:
            self.min_err = mse
            self.min_err_test_k = k
            self.min_err_test_params = self.params
        elif mse < self.min_err:
            self.min_err = mse
            self.min_err_test_k = k
            self.min_err_test_params = self.params
        return epoch

    def visualise_fit(self, method, k, epoch, output_path=None, errors=None):
        x_train = self.train_data[:,0]
        y_train = self.train_data[:,1]
        x_axis = np.linspace(np.min(x_train), np.max(x_train), 100)
        y_axis = np.zeros(len(x_axis))
        for i in range(len(x_axis)):
            y_axis[i] = self.func(x_axis[i])
        fix, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].plot(x_axis, y_axis, color='black', label='Fitted curve')
        for i in range(len(x_train))[::3]:
            axes[0, 0].plot([x_train[i], x_train[i]], [y_train[i], self.func(x_train[i])], color="blue", linewidth=0.5)
        axes[0, 0].scatter(x_train, y_train, color='red', label='Train', s=10)

        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")
        axes[0, 0].set_title("Scatter plot of train split with fitted curve")
        axes[0, 0].legend()

        errors = np.array(errors)
        x_axis = np.arange(0, epoch + 1)
        axes[0, 1].plot(x_axis, errors[:,0], color='red', label='MSE')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].set_title("MSE for Train Set")

        axes[1, 0].plot([0, epoch], [self.var, self.var], color='#AAAAFF', label='Variance of all data')
        axes[1, 0].plot(x_axis, errors[:,1], color='#0000FF', label='Var')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].set_title("Var for Train Set")
        axes[1, 0].legend()

        axes[1, 1].plot([0, epoch], [self.sd, self.sd], color='#AAFFAA', label='SD of all data')
        axes[1, 1].plot(x_axis, errors[:,2], color="#00FF00", label="SD")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].set_title("SD for Train Set")
        axes[1, 1].legend()

        if method == "show":
            plt.show()
        if method == "save in arr":
            if output_path is None:
                raise ValueError("Output path has to be provided to save the image")
            
            # Trying to save the plot directly as a numpy array was creating issues, so took easy way out
            # by first storing the image and then immediately reading it and storing it in a list
            plt.savefig(output_path, bbox_inches='tight')
            img = Image.open(output_path)
            self.img_arr.append(img)
            os.remove(output_path)
        if method == "save":
            plt.savefig(output_path, bbox_inches='tight')
        
        plt.close()

    def get_metrics(self, split):
        split = split.strip().lower()
        x = None
        y = None
        if split == "train":
            x = self.train_data[:,0]
            y = self.train_data[:,1]
        elif split == "validation":
            x = self.validation_data[:,0]
            y = self.validation_data[:,1]
        elif split == "test":
            x = self.test_data[:,0]
            y = self.test_data[:,1]
        y_pred = np.zeros(len(x))
        for i in range(len(x)):
            y_pred[i] = self.func(x[i])
        mse = round(self.MSE(y, y_pred), 4)
        var = round(self.variance(y_pred), 4)
        sd = round(self.standard_deviation(y_pred), 4)
        return mse, var, sd
    
    def save_best_model(self, path):
        with open(path, 'w') as f:
            final_string = ""
            for param in self.params:
                final_string += str(param) + ","
            final_string = final_string[:-1]
            f.write(final_string)

    def animate(self, output_path):
        self.img_arr[0].save(output_path,
               save_all=True,
               append_images=self.img_arr[1:],
               duration=0.1,
               loop=0,
               optimize=True)
