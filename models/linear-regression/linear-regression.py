import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

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
    
    def load_model(self, path):
        with open(path, "r") as f:
            params = f.read().split(",")
            self.k = len(params) - 1
            self.params = np.array([float(param) for param in params])
    
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
        self.params = np.zeros(k + 1)
        for epoch in range(epochs):
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
            self.visualise_train_and_fitted_curve("save", k, epoch)
        mse, var, sd = self.get_metrics("test")
        if self.min_err is None:
            self.min_err = mse
            self.min_err_test_k = k
            self.min_err_test_params = self.params
        elif mse < self.min_err:
            self.min_err = mse
            self.min_err_test_k = k
            self.min_err_test_params = self.params

    def visualise_train_and_fitted_curve(self, method, k, epoch):
        x_train = self.train_data[:,0]
        y_train = self.train_data[:,1]
        # x_axis = np.linspace(np.min(x_train), np.max(x_train), len(y_train))
        x_axis = sorted(x_train)
        y_axis = np.random.rand(len(x_axis))
        for i in range(len(x_axis)):
            y_axis[i] = self.func(x_axis[i])
        plt.plot(x_axis, y_axis, color='black', label='Fitted curve')
        for i in range(len(x_axis))[::3]:
            plt.plot([x_train[i], x_train[i]], [y_train[i], self.func(x_train[i])], color="blue", linewidth=0.5)
        plt.scatter(x_train, y_train, color='red', label='Train', s=10)
        # Get the current axis limits
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()

        # Set coordinates for the text
        text_x = xlim[1] + 0.1 * (xlim[1] - xlim[0])  # 10% to the right of the graph
        text_y_top = ylim[1]  # Align the top text near the top of the graph
        text_y_bottom = text_y_top - 0.1 * (ylim[1] - ylim[0])  # Below the top text

        # Add the text to the right of the graph, one below the other
        plt.text(text_x, text_y_top, f'k = {k}', fontsize=12, color='red', va='top')
        plt.text(text_x, text_y_bottom, f'Epoch = {epoch}', fontsize=12, color='red', va='top')
        plt.text(text_x, text_y_bottom, f'.                    ', fontsize=12, color='red', va='top')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatter plot of train split with fitted curve")
        plt.legend()
        if method == "show":
            plt.show()
        if method == "save":
            if k == None or epoch == None:
                raise("k and epoch must be provided for saving the image")
            plt.savefig(f'../../assignments/1/figures/train_fitted_curve_{k}_{epoch}.png', bbox_inches='tight')
        plt.clf()

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

    def animate(self, k, n_epochs):
        frames = []
        for i in range(n_epochs):
            img_path = f"../../assignments/1/figures/train_fitted_curve_{k}_{i}.png"
            img = Image.open(img_path)
            # img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))
            frames.append(img)

        output_gif_path = f"../../assignments/1/figures/animation_degree_{k}.gif"
        frames[0].save(output_gif_path,
               save_all=True,
               append_images=frames[1:],
               duration=0.1,
               loop=0,
               optimize=True)

        for i in range(n_epochs):
            # if i != n_epochs - 1:
            img_path = f"../../assignments/1/figures/train_fitted_curve_{k}_{i}.png"
            os.remove(img_path)
            # else:
            #     os.rename(img_path, f"../../assignments/1/figures/final_train_fitted_curve_{k}.png")

# Remove this part and write it where you want to run the code
# This is just for testing
if __name__ == "__main__":
    lr = 1
    n_epochs = 100
    max_k = 10
    data_path = "../../data/external/linreg.csv"
    data = np.genfromtxt(data_path, delimiter=',', skip_header=True)
    linreg = Regression()
    # linreg.load_model("./best_model_params.txt")
    linreg.load_data(data)
    linreg.shuffle_data()
    linreg.split_data(80, 10, 10)
    # linreg.visualise_split()

    train_metrics = list()
    test_metrics = list()
    for k in range(1, max_k + 1):
        linreg.fit(k, lr, n_epochs)
        mse_train, var_train, sd_train = linreg.get_metrics("train")
        train_metrics.append([mse_train, var_train, sd_train])
        print(f"Degree: {k}\nTraining set: MSE: {mse_train}, Variance: {var_train}, Standard Deviation {sd_train}")
        mse_test, var_test, sd_test = linreg.get_metrics("test")
        test_metrics.append([mse_test, var_test, sd_test])
        print(f"Test set: MSE: {mse_test}, Variance: {var_test}, Standard Deviation: {sd_test}")
        print()

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
    # plt.show()

    plt.plot(x_axis, test_metrics[:,0], color='red', label='MSE')
    plt.plot(x_axis, test_metrics[:,1], color='blue', label='Variance')
    plt.plot(x_axis, test_metrics[:,2], color="green", label="Std Dev")
    plt.xlabel("Degree")
    plt.ylabel("Value")
    plt.title("MSE, Bias and Var Plot for Test Set")
    plt.legend()
    # plt.show()
    linreg.save_best_model("best_model_params.txt")
    for i in range(1, max_k + 1):
        linreg.animate(i, n_epochs)
    # linreg.visualise_train_and_fitted_curve("final_train_fitted_curve")
