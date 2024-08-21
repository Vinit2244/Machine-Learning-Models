import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

'''
    Since we have changed the base path, make sure to provide path names from the root of the project
'''

from models.linear_regression.linear_regression import Regression
from models.knn.knn import KNN

def get_index(header, val):
    return header.index(val)

def preprocess(data):
    # Renaming the first unnamed column as index
    data = data.rename(columns={'Unnamed: 0': 'index'})

    # Remove exact duplicates - if track id is same, the song is exactly the same
    data = data.drop_duplicates("track_id")

    # Remove duplicate songs in multiple albums
    data = data.drop_duplicates("track_name")

    # Removing unnecessary columns
    unnecessary_cols = ["index", "track_id", "album_name"]
    data = data.drop(columns=unnecessary_cols)

    # Renaming all False as 0 and True as 1 in explicit column
    data['explicit'] = data['explicit'].replace({'False': '0', 'True': '1'})
    return data

def load_data():
    # Some data points have commas in them, so we need to use quotechar to read the file
    # Source: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    try:
        data = data = pd.read_csv("../../interim/preprocessed_spotify.csv", quotechar='"')
    except FileNotFoundError:
        data = pd.read_csv("../../data/external/spotify.csv", quotechar='"')
        data = preprocess(data)
        data.to_csv('../../data/interim/preprocessed_spotify.csv', index=False)
    return data

def plot_train_test_metrics(x, y):
    fig, axes = plt.subplots(1, len(y), figsize=(10, 5))
    idx = 0
    max_y = 0
    for key, value in y.items():
        max_mse = np.max(value[:, 0])
        max_var = np.max(value[:, 1])
        max_sd = np.max(value[:, 2])
        max_y = max(max_y, max_mse, max_var, max_sd)
        axes[idx].plot(x, value[:, 0], color='red', label='MSE')
        axes[idx].plot(x, value[:, 1], color='blue', label='Variance')
        axes[idx].plot(x, value[:, 2], color="green", label="Std Dev")
        axes[idx].set_xlabel("Degree")
        axes[idx].set_ylabel("Value")
        axes[idx].set_title(f"MSE, Bias and Var Plot for {key.capitalize()} Set")
        axes[idx].legend()
        idx += 1
    # Setting the y-axis limits to be the same for all plots for better comparison (same range)
    for i in range(idx):
        axes[i].set_ylim(0, max_y)
    plt.tight_layout()
    plt.show()

def analyse_data():
    data = load_data()

def knn():
    pass

def linear_regression(data):
    # Linear Regression Hyperparameters
    lr = 0.05
    diff_threshold = 0.0003
    max_k = 5
    seed = 10
    linreg = Regression()
    # linreg.load_model("./assignments/1/best_model_params.txt")
    linreg.load_data(data)
    linreg.shuffle_data()
    linreg.split_data(80, 10, 10)
    linreg.visualise_split()

    train_metrics = np.zeros((max_k, 3))
    test_metrics = np.zeros((max_k, 3))
    for k in range(1, max_k + 1):
        # Training the model
        converged_epoch = linreg.fit(k, lr, diff_threshold, save_epoch_imgs=True, seed=seed, max_epochs=100)
        # Train Metrics
        print("Degree:", k)
        print("Seed:", seed)
        print("Converged at epoch:", converged_epoch)
        mse_train, var_train, sd_train = linreg.get_metrics("train")
        train_metrics[k - 1] = np.array([mse_train, var_train, sd_train])
        print(f"Training set: MSE: {mse_train}, Variance: {var_train}, Standard Deviation {sd_train}")
        # Test Metrics
        mse_test, var_test, sd_test = linreg.get_metrics("test")
        test_metrics[k - 1] = np.array([mse_test, var_test, sd_test])
        print(f"Test set: MSE: {mse_test}, Variance: {var_test}, Standard Deviation: {sd_test}")
        print()
        linreg.animate(f"./assignments/1/figures/lin_reg_animation_{k}.gif")

    plot_train_test_metrics([i for i in range(1, max_k + 1)], {"train": train_metrics, "test": test_metrics})
    
    # linreg.save_best_model("./assignments/1/best_model_params.txt")

def l1_linear_regression(data):
    # L1 Regularization
    linreg_l1 = Regression(regularization_method="l1", lamda=0.1)
    linreg_l1.load_data(data)
    linreg_l1.shuffle_data()
    linreg_l1.split_data(80, 10, 10)
    linreg_l1.visualise_split()

    train_metrics = list()
    test_metrics = list()
    
    for k in range(1, 21):
        linreg_l1.fit(k, lr=1, epochs=100)
        mse_train, var_train, sd_train = linreg_l1.get_metrics("train")
        train_metrics.append([mse_train, var_train, sd_train])
        print(f"Degree: {k}\nTraining set: MSE: {mse_train}, Variance: {var_train}, Standard Deviation {sd_train}")
        mse_test, var_test, sd_test = linreg_l1.get_metrics("test")
        test_metrics.append([mse_test, var_test, sd_test])
        print(f"Test set: MSE: {mse_test}, Variance: {var_test}, Standard Deviation: {sd_test}")
        print()
        linreg_l1.visualise_train_and_fitted_curve("save", k, "l1")
    
    train_metrics = np.array(train_metrics)
    test_metrics = np.array(test_metrics)
    x_axis = [i for i in range(1, 21)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(x_axis, train_metrics[:,0], color='red', label='MSE')
    axes[0].plot(x_axis, train_metrics[:,1], color='blue', label='Variance')
    axes[0].plot(x_axis, train_metrics[:,2], color="green", label="Std Dev")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Value")
    axes[0].set_title("MSE, Bias and Var Plot for Train Set")
    axes[0].legend()

    axes[1].plot(x_axis, test_metrics[:,0], color='red', label='MSE')
    axes[1].plot(x_axis, test_metrics[:,1], color='blue', label='Variance')
    axes[1].plot(x_axis, test_metrics[:,2], color="green", label="Std Dev")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Value")
    axes[1].set_title("MSE, Bias and Var Plot for Test Set")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def l2_linear_regression(data):
    # L2 Regularization
    linreg_l2 = Regression(regularization_method="l2", lamda=0.1)
    linreg_l2.load_data(data)
    linreg_l2.shuffle_data()
    linreg_l2.split_data(80, 10, 10)
    linreg_l2.visualise_split()

    train_metrics = list()
    test_metrics = list()

    for k in range(1, 21):
        linreg_l2.fit(k, lr=1, epochs=100)
        mse_train, var_train, sd_train = linreg_l2.get_metrics("train")
        train_metrics.append([mse_train, var_train, sd_train])
        print(f"Degree: {k}\nTraining set: MSE: {mse_train}, Variance: {var_train}, Standard Deviation {sd_train}")
        mse_test, var_test, sd_test = linreg_l2.get_metrics("test")
        test_metrics.append([mse_test, var_test, sd_test])
        print(f"Test set: MSE: {mse_test}, Variance: {var_test}, Standard Deviation: {sd_test}")
        print()
        linreg_l2.visualise_train_and_fitted_curve("save", k, "l2")

    train_metrics = np.array(train_metrics)
    test_metrics = np.array(test_metrics)
    x_axis = [i for i in range(1, 21)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(x_axis, train_metrics[:,0], color='red', label='MSE')
    axes[0].plot(x_axis, train_metrics[:,1], color='blue', label='Variance')
    axes[0].plot(x_axis, train_metrics[:,2], color="green", label="Std Dev")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Value")
    axes[0].set_title("MSE, Bias and Var Plot for Train Set")
    axes[0].legend()

    axes[1].plot(x_axis, test_metrics[:,0], color='red', label='MSE')
    axes[1].plot(x_axis, test_metrics[:,1], color='blue', label='Variance')
    axes[1].plot(x_axis, test_metrics[:,2], color="green", label="Std Dev")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Value")
    axes[1].set_title("MSE, Bias and Var Plot for Test Set")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    '''
        Data Analysis Part
    '''
    # analyse_data()

    '''
        KNN Part
    '''
    # knn()

    '''
        Linear Regression Part
    '''
    data_path = "./data/external/linreg.csv"
    data = np.genfromtxt(data_path, delimiter=',', skip_header=True)
    linear_regression(data)
    data_regularisation_path = "./data/external/regularisation.csv"
    data_regularisation = np.genfromtxt(data_regularisation_path, delimiter=',', skip_header=True)
    # l1_linear_regression(data_regularisation)
    # l2_linear_regression(data_regularisation)