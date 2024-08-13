import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def __init__(self):
        self.params = None # Stored in increasing order of degree
        self.error = 0
        self.data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.min_err_test_k = None
        self.min_err_test_params = None
        self.min_err = None
    
    def load_data(self, data):
        self.data = data
    
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

    def MSE(self, y_true, y_pred):
        mse = 0
        for i in range(len(y_true)):
            mse += (y_true[i] - y_pred[i]) ** 2
        mse /= len(y_true)
        return mse
    
    def variance(self, y_true, y_pred):
        mse = self.MSE(y_true, y_pred)
        var = 0
        for y in y_pred:
            var += (y - mse) ** 2
        return var / len(y_pred)
    
    def standard_deviation(self, y_true, y_pred):
        var = self.variance(y_true, y_pred)
        return np.sqrt(var)

    def func(self, x):
        y_pred = 0
        x_powered = 1
        for i in range(len(self.params)):
            y_pred += self.params[i] * x_powered
            x_powered *= x
        return y_pred
    
    def fit(self, k, lr, epochs):
        # Note here for the value of y intercept we can also use the y value of the first point for faster convergence
        self.params = np.random.rand(k + 1)
        for _ in range(epochs):
            x = self.train_data[:,0]
            y = self.train_data[:,1]
            y_pred = np.zeros(len(x))
            for i in range(len(x)):
                y_pred[i] = self.func(x[i])
            y_error = y_pred - y
            for i in range(k + 1):
                final_sum = 0
                for j in range(len(y_error)):
                    final_sum += y_error[j] * (x[j] ** i)
                self.params[i] = self.params[i] - lr * final_sum / len(y_error)
        mse, var, sd = self.get_metrics("test")
        if self.min_err is None:
            self.min_err = mse
            self.min_err_test_k = k
            self.min_err_test_params = self.params
        elif mse < self.min_err:
            self.min_err = mse
            self.min_err_test_k = k
            self.min_err_test_params = self.params

    # Function for testing
    def visualise_train_and_fitted_curve(self):
        x_train = self.train_data[:,0]
        y_train = self.train_data[:,1]
        x_axis = np.linspace(np.min(x_train), np.max(x_train), 100)
        y_axis = np.random.rand(len(x_axis))
        for i in range(len(x_axis)):
            y_axis[i] = self.func(x_axis[i])
        plt.scatter(x_train, y_train, color='red', label='Train', s=10)
        plt.plot(x_axis, y_axis, color='black', label='Fitted curve')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatter plot of train split with fitted curve")
        plt.legend()
        plt.show()

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
        var = round(self.variance(y, y_pred), 4)
        sd = round(self.standard_deviation(y, y_pred), 4)
        return mse, var, sd
    
    def save_best_model(self, path):
        with open(path, 'w') as f:
            final_string = ""
            for param in self.params:
                final_string += str(param) + ","
            final_string = final_string[:-1]
            f.write(final_string)

# Remove this part and write it where you want to run the code
# This is just for testing
if __name__ == "__main__":
    lr = 0.01
    n_epochs = 1000
    max_k = 10
    data_path = "../../data/external/linreg.csv"
    data = np.genfromtxt(data_path, delimiter=',', skip_header=True)
    linreg = Regression()
    linreg.load_data(data)
    linreg.shuffle_data()
    linreg.split_data(80, 10, 10)
    linreg.visualise_split()

    train_metrics = list()
    test_metrics = list()
    for k in range(1, max_k + 1):
        linreg.fit(k, lr, n_epochs)
        mse_train, var_train, sd_train = linreg.get_metrics("train")
        train_metrics.append([mse_train, var_train, sd_train])
        print(f"Degree: {k},\nTraining set: MSE: {mse_train}, Variance: {var_train}, Standard Deviation {sd_train}")
        print()
        mse_test, var_test, sd_test = linreg.get_metrics("test")
        test_metrics.append([mse_test, var_test, sd_test])
        print(f"Test set: MSE: {mse_test}, Variance: {var_test}, Standard Deviation: {sd_test}")

    train_metrics = np.array(train_metrics)
    test_metrics = np.array(test_metrics)
    x_axis = [i for i in range(1, max_k + 1)]
    plt.plot(x_axis, train_metrics[:,0], color='red', label='MSE')
    plt.plot(x_axis, train_metrics[:,1], color='blue', label='Variance')
    plt.plot(x_axis, train_metrics[:,2], color="green", label="Std Dev")
    plt.xlabel("Degree")
    plt.ylabel("Value")
    plt.title("MSE, Bias and Var Plot for Train Set")
    plt.legend()
    plt.show()

    plt.plot(x_axis, test_metrics[:,0], color='red', label='MSE')
    plt.plot(x_axis, test_metrics[:,1], color='blue', label='Variance')
    plt.plot(x_axis, test_metrics[:,2], color="green", label="Std Dev")
    plt.xlabel("Degree")
    plt.ylabel("Value")
    plt.title("MSE, Bias and Var Plot for Test Set")
    plt.legend()
    plt.show()
    linreg.visualise_train_and_fitted_curve()
    linreg.save_best_model("best_model_params.txt")




'''Write function to load parameters from given file path and then load those params into the model'''