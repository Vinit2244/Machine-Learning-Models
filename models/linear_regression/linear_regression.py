import numpy as np                  # For array manipulation
import matplotlib.pyplot as plt     # For plotting the data
import os                           # For removing the temporary image file
from PIL import Image               # For reading the image file
from performance_measures.performance_measures import PerformanceMetrics # MSE, Variance, Standard Deviation

# Class to perform linear regression, it inherits all the performance metrics from the PerformanceMetrics class
class Regression(PerformanceMetrics):
    def __init__(self, regularization_method=None, lamda=0):
        super().__init__()
        self.params = None                                      # Stored in increasing order of degree
        self.lamda = lamda                                      # Regularization parameter
        self.regularization_method = regularization_method      # Regularization method
        self.img_arr = list()                                   # List to store the images for creating GIF
        self.train_data = None                                  # Training data
        self.test_data = None                                   # Testing data
        self.validation_data = None                             # Validation data
        self.var = None                                         # Variance of all data
    
    # Returns the parameters of the model after training
    def get_params(self):
        return self.params

    # Function to load the parameters of the model from a file
    def load_model(self, path):
        with open(path, "r") as f:
            params = f.read().split(",")
            self.k = len(params) - 1
            self.params = np.array([float(param) for param in params])

    # Function to load the data into the model
    def load_train_test_validation_data(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        train_var = self.variance(self.train_data[:,1])
        test_var = self.variance(self.test_data[:,1])
        validation_var = self.variance(self.validation_data[:,1])

        train_sd = self.standard_deviation(self.train_data[:,1])
        test_sd = self.standard_deviation(self.test_data[:,1])
        validation_sd = self.standard_deviation(self.validation_data[:,1])

        print(f"""Metrics for all three splits:\n
                  Train:      Variance: {train_var}, Standard Deviation: {train_sd}\n
                  Validation: Variance: {validation_var}, Standard Deviation: {validation_sd}\n
                  Test:       Variance: {test_var}, Standard Deviation: {test_sd}\n""")

    # Function to calculate the value of the polynomial at a given x
    def func(self, x):
        if self.params is None:
            raise ValueError("Model has not been initialised properly yet")
        y_pred = 0
        x_powered = 1
        for i in range(len(self.params)):
            y_pred += self.params[i] * x_powered
            x_powered *= x
        return y_pred
    
    # Fits the model to the data
    def fit(self, k, lr, threshold, save_epoch_imgs=False, seed=42, max_epochs=np.inf):
        if save_epoch_imgs:
            self.img_arr = list()

        # Note here for the value of y intercept we can also use the y value of the first point for faster convergence
        np.random.seed(seed)
        self.params = np.random.rand(k + 1)
        errors = list()
        epoch = -1

        while (len(errors) < 2) or (abs(errors[-1][0] - errors[-2][0]) > threshold) and (epoch < max_epochs):
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
                    # Calculating the derivative of the loss function
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
                # Gradient Descent
                self.params[i] = self.params[i] - lr * final_sum / len(y_error)
            if save_epoch_imgs:
                self.visualise_fit(self.train_data, "save in arr", epoch, "temp.png", errors)
        return epoch

    # Predicts the output for the given data
    def predict(self, data):
        y_pred = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            y_pred[i] = self.func(data[i])
        return y_pred

    # Function to visualise the data and the best fit line calculated
    def visualise_fit(self, data, method, epoch=None, output_path=None, errors=None):
        x_data = data[:,0]
        y_data = data[:,1]
        x_axis = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_axis = np.zeros(len(x_axis))

        for idx, x_val in enumerate(x_axis):
            y_axis[idx] = self.func(x_val)

        if epoch != None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"Epoch: {epoch}")
            axes[0, 0].plot(x_axis, y_axis, color='black', label='Fitted curve')
            for i in range(len(x_data))[::3]:
                axes[0, 0].plot([x_data[i], x_data[i]], [y_data[i], self.func(x_data[i])], color="blue", linewidth=0.5)
            axes[0, 0].scatter(x_data, y_data, color='red', label='Data', s=10)

            axes[0, 0].set_xlabel("X")
            axes[0, 0].set_ylabel("Y")
            axes[0, 0].set_title("Scatter plot of data with fitted curve")
            axes[0, 0].legend()

            errors = np.array(errors)
            x_axis = np.arange(0, epoch + 1)
            axes[0, 1].plot(x_axis, errors[:,0], color='red', label='MSE')
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Value")
            axes[0, 1].set_title("MSE")

            data_var = self.variance(y_data)
            axes[1, 0].plot([0, epoch], [data_var, data_var], color='#AAAAFF', label='Variance of all data', linestyle='--')
            axes[1, 0].plot(x_axis, errors[:,1], color='#0000FF', label='Var')
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].set_title("Var")
            axes[1, 0].legend()

            data_sd = self.standard_deviation(y_data)
            axes[1, 1].plot([0, epoch], [data_sd, data_sd], color='#AAFFAA', label='SD of all data', linestyle='--')
            axes[1, 1].plot(x_axis, errors[:,2], color="#00FF00", label="SD")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Value")
            axes[1, 1].set_title("SD")
            axes[1, 1].legend()
        else:
            fix, axis = plt.subplots(1, 1, figsize=(5, 5))
            axis.plot(x_axis, y_axis, color='black', label='Fitted curve')
            for i in range(len(x_data))[::3]:
                axis.plot([x_data[i], x_data[i]], [y_data[i], self.func(x_data[i])], color="blue", linewidth=0.5)
            axis.scatter(x_data, y_data, color='red', label='Train', s=10)

            axis.set_xlabel("X")
            axis.set_ylabel("Y")
            axis.set_title("Scatter plot of data with fitted curve")
            axis.legend()

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

    # Returns the metrics for the given split
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
        y_pred = self.predict(x)
        mse = round(self.MSE(y, y_pred), 4)
        var = round(self.variance(y_pred), 4)
        sd = round(self.standard_deviation(y_pred), 4)
        return mse, var, sd
    
    # Saves the trained model's parameters to a file
    def save_model(self, path):
        with open(path, 'w') as f:
            final_string = ""
            for param in self.params:
                final_string += str(param) + ","
            final_string = final_string[:-1]
            f.write(final_string)

    # Creates GIF out of the images stored in the img_arr
    def animate_training(self, output_path):
        if self.img_arr is None:
            raise ValueError("No images have been stored yet, run the fit function with save_epoch_imgs=True")
        self.img_arr[0].save(output_path,
               save_all=True,
               append_images=self.img_arr[1:],
               duration=0.1,
               loop=0,
               optimize=True)
