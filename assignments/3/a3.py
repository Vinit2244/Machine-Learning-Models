import os
import sys
import numpy as np
import pandas as pd
import prettytable as pt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import wandb
from prettytable import PrettyTable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.knn.knn import KNN
from models.MLP.MLP import MLP
from models.AutoEncoder.AutoEncoder import AutoEncoder
from performance_measures.performance_measures import PerformanceMetrics

USE_WANDB = False

TRAIN_PERCENT = 70
TEST_PERCENT = 15
VAL_PERCENT = 15

optimisers = ["batch", "sgd", "mini_batch", None]
learning_rates = [0.001, 0.01, 0.1, 1]
activation_funcs = ["sigmoid", "tanh", "relu", "linear"]
epochs = 100
batch_size_arr = [16, 32, 64, 128]

# Colors for printing for better readability
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
"""
Testing code:

X = np.array(
    [np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9])]
)

regressor_model = MLP_Regression(
    n_ip=3, 
    neurons_per_hidden_layer=[4], 
    n_op=1,
    learning_rate=0.01,
    activation_func="linear",
    optimiser="batch",
    epochs=1,
    batch_size=10
)

prediction = regressor_model.predict(X)
print(prediction)
"""

# Returns the train, test, val split (in that order)
def split_data(data, train_percent, test_percent, val_percent=None, shuffle=True, seed=42):
    """
        Returns the train, test, val split (in that order)

        Parameters
        ==========
            data[n_points, n_features] (numpy array): The data to be split
            train_percent (int): Percentage of data for training
            test_percent (int): Percentage of data for testing
            val_percent (int): Percentage of data for validation
            shuffle (bool): Whether to shuffle the data before splitting
            seed (int): Seed for shuffling

        Returns
        =======
            train_data[n_train, n_features] (numpy array): Data for training
            test_data[n_test, n_features] (numpy array): Data for testing
            val_data[n_val, n_features] (numpy array): Data for validation
    """
    if train_percent + test_percent > 100:
        raise ValueError("Train and Test percentages should not sum to more than 100")
    
    if val_percent is None:
        val_percent = 100 - train_percent - test_percent
    elif train_percent + test_percent + val_percent > 100:
        raise ValueError("Train, Test and Validation percentages should not sum to more than 100")
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(data)
    
    # Calculating the number of data points for each split
    n_train = int(train_percent * len(data) / 100)
    n_test = int(test_percent * len(data) / 100)
    n_val = int(val_percent * len(data) / 100)

    train_data = data[:n_train]
    test_data = data[n_train:n_train + n_test]
    val_data = data[n_train + n_test:]

    return train_data, test_data, val_data

def describe_dataset(data, headers):
    mean, sd, min_vals, max_vals = np.mean(data, axis=0), np.std(data, axis=0), np.min(data, axis=0), np.max(data, axis=0)
    print(f"{GREEN}Mean, Standard Deviation, Min, Max values of the dataset features:{RESET}")
    attr_table = pt.PrettyTable()
    attr_table.field_names = ["Attribute", "Mean", "Standard Deviation", "Min", "Max"]
    for i in range(headers.shape[0]):
        attr_table.add_row([headers[i], mean[i], sd[i], min_vals[i], max_vals[i]])
    print(attr_table, end="\n\n")

def plot_feature_distribution_histograms(data, headers, save_as=None, n_rows=3, n_cols=4, n_range=12):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    colors = sns.color_palette("tab20", n_range)

    # Plot histogram for each feature
    for i in range(n_range):
        ax = axes[i//n_cols, i%n_cols]
        if headers[i] == "quality":
            sns.histplot(data[:, i], bins=10, ax=ax, color=colors[i], kde=True, discrete=True)
        else:
            sns.histplot(data[:, i], bins=20, ax=ax, color=colors[i], kde=True)
        
        ax.set_title(headers[i], fontsize=16, fontweight='bold', color='darkblue')
        ax.set_xlabel(headers[i], fontsize=12, color='darkgreen')
        ax.set_ylabel("Frequency", fontsize=12, color='darkred')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Histograms of Data Features", fontsize=22, fontweight='bold', color='darkblue', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"./assignments/3/figures/dataset_analysis/histograms_{save_as}.png")
    plt.close()

def normalise_and_standardise(data):
    # Normalizing the data
    min_max_scaler = MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(data)

    # Standardizing the data
    standard_scaler = StandardScaler()
    standardized_data = standard_scaler.fit_transform(data)

    return normalized_data, standardized_data

# Preprocess the data
def preprocess(data):
    # Renaming the first unnamed column as index
    data = data.rename(columns={'Unnamed: 0': 'index'})

    # Remove exact duplicates - if track id is same, the song is exactly the same
    data = data.drop_duplicates("track_id")

    # Remove duplicate songs in multiple albums
    data = data.drop_duplicates("track_name")

    # Removing unnecessary columns
    # Removing all the categorical columns as well except the genre
    unnecessary_cols = ["index", "track_id", "album_name", "explicit", "artists", "track_name"]
    data = data.drop(columns=unnecessary_cols)

    # Z-score normalization
    cols_to_normalise = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]
    for col in cols_to_normalise:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std

    data.iloc[:, :-1] = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    return data

def one_hot_encode_single_class(data, n_classes):
    one_hot_data = np.zeros((data.shape[0], n_classes))
    for i in range(data.shape[0]):
        one_hot_data[i][int(data[i]) - 1] = 1
    return one_hot_data

def one_hot_encode_multi_class(data):
    unique_labels = list()
    for label_row in data:
        for label in label_row:
            if label not in unique_labels:
                unique_labels.append(label)
    unique_labels.sort()
    n_classes = len(unique_labels)
    one_hot_data = np.zeros((len(data), n_classes))
    for i in range(len(data)):
        for label in data[i]:
            one_hot_data[i][unique_labels.index(label)] = 1
    return one_hot_data

def label_encode(data):
    unique_labels = list()
    for label in data:
        if label not in unique_labels:
            unique_labels.append(label)
    unique_labels.sort()
    for i in range(len(data)):
        data[i] = unique_labels.index(data[i])
    return data

def print_aprf1(model_type, lr, act_func, optimiser, epochs, batch_size, layers, model, X_test, y_test, test_p, test_r, test_f1):
    if model_type == "single_class_classifier":
        accuracy = round(model.compute_acc(y_test, model.forward_prop(X_test)[-1]) * 100, 3)
    elif model_type == "multi_class_classifier":
        accuracy = round(model.compute_acc(y_test, model.forward_prop(X_test)[-1], multi_class=True) * 100, 3)
    macro_precision = round(np.mean(test_p) * 100, 3)
    macro_recall = round(np.mean(test_r) * 100, 3)
    macro_f1 = round(np.mean(test_f1), 3)
    
    print(f"""
    {MAGENTA}Model Params:{RESET}
        Learning Rate: {GREEN}{lr}{RESET}
        Activation Function: {GREEN}{act_func}{RESET}
        Optimiser: {GREEN}{optimiser}{RESET}
        Epochs: {GREEN}{epochs}{RESET}
        Batch Size: {GREEN}{batch_size}{RESET}
        Layers: {GREEN}{layers}{RESET}

    {MAGENTA}Final Metrics:{RESET}
        Test Accuracy: {GREEN}{accuracy}%{RESET}

        Macro Test Precision: {GREEN}{macro_precision}%{RESET}

        Macro Test Recall: {GREEN}{macro_recall}%{RESET}

        Macro Test F1 Score: {GREEN}{macro_f1}{RESET}

        -----------------------------------------------------------

        Micro Test Precision: {GREEN}{[float(x) for x in test_p]}{RESET}

        Micro Test Recall: {GREEN}{[float(x) for x in test_r]}{RESET}

        Micro Test F1 Score: {GREEN}{[float(x) for x in test_f1]}{RESET}
    """)
    return [accuracy, macro_precision, macro_recall, macro_f1]

def print_mse_rmse_r2(lr, act_func, optimiser, epochs, batch_size, layers, mse_test, rmse_test, r2_test):
    print(f"""
    {MAGENTA}Model Params:{RESET}
        Learning Rate: {GREEN}{lr}{RESET}
        Activation Function: {GREEN}{act_func}{RESET}
        Optimiser: {GREEN}{optimiser}{RESET}
        Epochs: {GREEN}{epochs}{RESET}
        Batch Size: {GREEN}{batch_size}{RESET}
        Layers: {GREEN}{layers}{RESET}

    {MAGENTA}Final Metrics:{RESET}
        Test Mean Squared Error: {GREEN}{mse_test}{RESET}

        Test Root Mean Squared Error: {GREEN}{rmse_test}{RESET}

        Test R2 Score: {GREEN}{r2_test}{RESET}
    """)

def init_wandb(model_type, hyperparameters, layers):
    if model_type == "single_class_classifier":
        wandb.init(
            project="assignment3-MLPClassifier_SingleClass",
            name=f"{hyperparameters["optimiser"]}_{hyperparameters["learning_rate"]}_{hyperparameters["activation_func"]}_layers{"_".join([str(x) for x in layers])}",
            config=hyperparameters
        )
    elif model_type == "multi_class_classifier":
        wandb.init(
            project="assignment3-MLPClassifier_MultiClass",
            name=f"{hyperparameters["optimiser"]}_{hyperparameters["learning_rate"]}_{hyperparameters["activation_func"]}_layers{"_".join([str(x) for x in layers])}",
            config=hyperparameters
        )
    elif model_type == "regressor":
        wandb.init(
            project="assignment3-MLPRegressor",
            name=f"{hyperparameters["optimiser"]}_{hyperparameters["learning_rate"]}_{hyperparameters["activation_func"]}_layers{"_".join([str(x) for x in layers])}",
            config=hyperparameters
        )

def log_wandb(metrics_arr):
    # Logging metrics onto W&B
    for metric in metrics_arr:
        wandb.log(metric)
    wandb.finish()

def run_model(model_type, lr, act_func, optimiser, epochs, batch_size, layers, X_train, y_train, X_val, y_val, X_test, y_test, loss):
    hyperparameters = {
        "learning_rate": lr,
        "activation_func": act_func,
        "optimiser": optimiser,
        "epochs": epochs,
        "batch_size": batch_size,
        "layers": layers
    }

    if USE_WANDB:
        init_wandb(model_type, hyperparameters, layers)

    # Defining the model
    model = MLP(
                n_ip=layers[0], 
                neurons_per_hidden_layer=layers[1:-1], 
                n_op=layers[-1],
                learning_rate=lr,
                activation_func=act_func,
                optimiser=optimiser,
                epochs=epochs,
                batch_size=batch_size,
                loss=loss
                )

    # Logging training time
    start_time = time.time()
    # Training the model
    if model_type == "multi_class_classifier":
        metrics_arr = model.fit(X_train, y_train, X_val, y_val, multi_class=True)
    else:
        metrics_arr = model.fit(X_train, y_train, X_val, y_val)
    end_time = time.time()
    print(f"{GREEN}Model Training completed successfully in time {end_time - start_time} s!{RESET}\n")

    if USE_WANDB:
        log_wandb(metrics_arr)

    if model_type == "single_class_classifier" or model_type == "multi_class_classifier":
        # Saving the loss for each model so it can later be used in part 2.5
        train_loss_arr = np.zeros(epochs)
        val_loss_arr = np.zeros(epochs)
        for i in range(epochs):
            train_loss_arr[i] = metrics_arr[i]["loss_train"]
            val_loss_arr[i] = metrics_arr[i]["loss_val"]

        # Calculating precision, recall and f1 scores for each of the classes
        if model_type == "single_class_classifier":
            test_p, test_r, test_f1 = PerformanceMetrics().get_precision_recall_f1_onehot_single_class(y_test, model.predict(X_test))
        elif model_type == "multi_class_classifier":
            test_p, test_r, test_f1 = PerformanceMetrics().get_precision_recall_f1_onehot_multi_class(y_test, model.predict(X_test))

        # Calculating and printing the accuracy, macro precision, macro recall and macro f1 score
        aprf1 = print_aprf1(model_type, lr, act_func, optimiser, epochs, batch_size, layers, model, X_test, y_test, test_p, test_r, test_f1)

        # For printing a table of all the hyperparameters tested and their metrics
        table_row = [lr, act_func, optimiser, epochs, batch_size, layers, aprf1[0], aprf1[1], aprf1[2], aprf1[3]]
        
        to_return = {f"{optimiser}_{lr}_{act_func}_{batch_size}_{"_".join([str(x) for x in layers])}": (train_loss_arr, val_loss_arr)}, table_row, aprf1, model, hyperparameters

    elif model_type == "regressor":
        # Calculating precision, recall and f1 scores for each of the classes
        prediction = model.predict(X_test)
        mse_test = PerformanceMetrics().MSE(y_test, prediction)
        rmse_test = PerformanceMetrics().RMSE(y_test, prediction)
        r2_test = PerformanceMetrics().R2(y_test, prediction)

        # Calculating and printing the accuracy, macro precision, macro recall and macro f1 score
        print_mse_rmse_r2(lr, act_func, optimiser, epochs, batch_size, layers, mse_test, rmse_test, r2_test)

        # For printing a table of all the hyperparameters tested and their metrics
        table_row = [lr, act_func, optimiser, epochs, batch_size, layers, mse_test, rmse_test, r2_test]
        
        to_return = table_row, mse_test, model, hyperparameters

    return to_return

def hyperparam_tuning(model_type, n_ip, n_op, X_train, y_train, X_val, y_val, X_test, y_test):
    layers_arr = [[n_ip, 32, 32, n_op],
                  [n_ip, 32, 64, 32, 16, n_op],
                  [n_ip, 128, n_op],
                  [n_ip, 64, 64, n_op],
                  [n_ip, 15, 15, n_op]
                 ]
    
    hyperparameters_table_rows = list()

    best_model_params = {}
    best_model_metric = None
    best_model_metrics = None

    loss_dict = {}

    for layers in layers_arr:
        for lr in learning_rates:
            for act_func in activation_funcs:
                for optimiser in optimisers:
                    # For mini_batch optimiser we have to iterate over batch size array as well
                    if optimiser == "mini_batch":
                        iteration_arr_batch_size = batch_size_arr
                    else:
                        iteration_arr_batch_size = [None]
                    
                    for batch_size in iteration_arr_batch_size:
                        if model_type == "single_class_classifier" or model_type == "multi_class_classifier":
                            loss_d, table_row, aprf1, trained_model, hyperparameters = run_model(model_type, lr, act_func, optimiser, epochs, batch_size, layers, X_train, y_train, X_val, y_val, X_test, y_test, loss="cross_entropy")
                            loss_dict.update(loss_d)
                        
                            # Storing the best model parameters (best model is the one with highest accuracy)
                            if best_model_metric is None or best_model_metric < aprf1[0]:
                                best_model_metric = aprf1[0]
                                best_model_params = hyperparameters
                                trained_model.save_model(f"./best_model_{model_type}.txt")
                                # Storing accuracy, macro precision, macro recall, macro f1 and loss of the best model to be printed later
                                best_model_metrics = aprf1 + [trained_model.compute_loss(y_test, trained_model.predict(X_test))]

                        elif model_type == "regressor":
                            table_row, mse_test, trained_model, hyperparameters = run_model(model_type, lr, act_func, optimiser, epochs, batch_size, layers, X_train, y_train, X_val, y_val, X_test, y_test, loss="mse")
                            # Storing the best model parameters (best model is the one with lowest MSE)
                            if best_model_metric is None or best_model_metric > mse_test:
                                best_model_metric = mse_test
                                best_model_params = hyperparameters
                                trained_model.save_model(f"./best_model_{model_type}.txt")
                        
                        hyperparameters_table_rows.append(table_row)

    hyperparameters_table = PrettyTable()
    if model_type == "single class classifier":
        hyperparameters_table.field_names = ["Learning Rate", "Activation Function", "Optimiser", "Epochs", "Batch Size", "Layers", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Cross Entropy Loss"]
    elif model_type == "multi class classifier":
        hyperparameters_table.field_names = ["Learning Rate", "Activation Function", "Optimiser", "Epochs", "Batch Size", "Layers", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Hamming Loss"]
    elif model_type == "regressor":
        hyperparameters_table.field_names = ["Learning Rate", "Activation Function", "Optimiser", "Epochs", "Batch Size", "Layers", "MSE", "RMSE", "R2"]
    
    for row in hyperparameters_table_rows:
        hyperparameters_table.add_row(row)

    print(GREEN)
    print("Hyperparameters Table:")
    print(hyperparameters_table)
    print(RESET)

    print(f"{MAGENTA}Best Model Params:{RESET}")
    print(best_model_params)
    print()

    if model_type == "single_class_classifier":
        print(f"{MAGENTA}Best Model Metrics:{RESET}")
        print(f"Test Accuracy: {GREEN}{best_model_metrics[0]}{RESET}%")
        print(f"Test Macro Precision: {GREEN}{best_model_metrics[1]}{RESET}%")
        print(f"Test Macro Recall: {GREEN}{best_model_metrics[2]}{RESET}%")
        print(f"Test Macro F1: {GREEN}{best_model_metrics[3]}{RESET}")
        print(f"Test Loss: {GREEN}{best_model_metrics[4]}{RESET}")
        print()

    elif model_type == "multi_class_classifier":
        print(f"{MAGENTA}Best Model Metrics:{RESET}")
        print(f"Test Accuracy: {GREEN}{best_model_metrics[0]}{RESET}%")
        print(f"Test Macro Precision: {GREEN}{best_model_metrics[1]}{RESET}%")
        print(f"Test Macro Recall: {GREEN}{best_model_metrics[2]}{RESET}%")
        print(f"Test Macro F1: {GREEN}{best_model_metrics[3]}{RESET}")
        print(f"Test Hamming Loss: {GREEN}{best_model_metrics[4]}{RESET}")

    if model_type == "single_class_classifier":
        return best_model_params, loss_dict
    elif model_type == "multi_class_classifier":
        return best_model_params, loss_dict

def plot_effect_of_tuning(model_type, best_model_params, loss_dict):
    # Hyperparameters of the best model
    lr = best_model_params["learning_rate"]
    optimiser = best_model_params["optimiser"]
    epochs = best_model_params["epochs"]
    batch_size = best_model_params["batch_size"]
    layers = best_model_params["layers"]
    act_func = best_model_params["activation_func"]

    # Analysing effect of non-linearity
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for idx, af in enumerate(activation_funcs):
        ax = axes[idx//2, idx%2]
        train_loss_per_epoch, val_loss_per_epoch = loss_dict[f"{optimiser}_{lr}_{af}_{batch_size}_{"_".join([str(x) for x in layers])}"]
        ax.plot(range(epochs), train_loss_per_epoch, label=f"Training Loss")
        ax.plot(range(epochs), val_loss_per_epoch, label=f"Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"activation function = {af}")
        ax.legend()
    plt.suptitle(f"lr={lr} op={optimiser} bs={batch_size} layers={"_".join([str(x) for x in layers])} and changing activation function")
    plt.savefig(f"./assignments/3/figures/effect_of_non_linearity_{model_type}.png")

    # Analysing effect of learning rates
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx//2, idx%2]
        train_loss_per_epoch, val_loss_per_epoch = loss_dict[f"{optimiser}_{lr}_{act_func}_{batch_size}_{"_".join([str(x) for x in layers])}"]
        ax.plot(range(epochs), train_loss_per_epoch, label=f"Training Loss")
        ax.plot(range(epochs), val_loss_per_epoch, label=f"Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"learning rate = {lr}")
        ax.legend()
    plt.suptitle(f"act_func={act_func} op={optimiser} bs={batch_size} layers={"_".join([str(x) for x in layers])} and changing learning rate")
    plt.savefig(f"./assignments/3/figures/effect_of_learning_rate_{model_type}.png")

    # Analysing effect of batch size only if the method is mini-batch
    if optimiser == "mini_batch":
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for idx, batch_size in enumerate(batch_size_arr):
            ax = axes[idx//2, idx%2]
            train_loss_per_epoch, val_loss_per_epoch = loss_dict[f"{optimiser}_{lr}_{act_func}_{batch_size}_{"_".join([str(x) for x in layers])}"]
            ax.plot(range(epochs), train_loss_per_epoch, label=f"Training Loss")
            ax.plot(range(epochs), val_loss_per_epoch, label=f"Validation Loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title(f"batch size = {batch_size}")
            ax.legend()
        plt.suptitle(f"act_func={act_func} op={optimiser} lr={lr} layers={"_".join([str(x) for x in layers])} and changing batch size")
        plt.savefig(f"./assignments/3/figures/effect_of_batch_size_{model_type}.png")
    else:
        print(f"We cannot do analysis for different batch sizes for {optimiser} optimiser")

def MLPSingleClassClassification():
    # Reading the dataset
    data_path = "./data/external/WineQT.csv"
    df = pd.read_csv(data_path)

    # Handle missing values by filling in mean values (No missing values found in WineQT dataset)
    # df.fillna(df.mean(), inplace=True)

    headers = df.columns.to_numpy()
    data = df.to_numpy()

    # Dropping the last column and splitting the data into features and labels
    headers = headers[:-1]
    data = data[:, :-1]

    # =================================================================
    #               2.1 Dataset Analysis and Preprocessing 
    # =================================================================
    describe_dataset(data, headers)

    plot_feature_distribution_histograms(data, headers, save_as="WineQT")
    print(f"{GREEN}Histograms of data features saved successfully for WineQT Dataset!{RESET}\n")

    train_data, test_data, val_data = split_data(data, train_percent=TRAIN_PERCENT, test_percent=TEST_PERCENT, val_percent=VAL_PERCENT)
    print(f"{GREEN}Partitioned the WineQT dataset in {TRAIN_PERCENT}:{TEST_PERCENT}:{VAL_PERCENT}!{RESET}\n")
    
    n_classes = len(np.unique(train_data[:, -1]))

    train_feat, y_train = train_data[:, :-1], train_data[:, -1]
    test_feat, y_test = test_data[:, :-1], test_data[:, -1]
    val_feat, y_val = val_data[:, :-1], val_data[:, -1]

    # Normalizing and Standardizing the data
    normalised_train_data, standardised_train_data = normalise_and_standardise(train_feat)
    normalised_test_data, standardised_test_data = normalise_and_standardise(test_feat)
    normalised_val_data, standardised_val_data = normalise_and_standardise(val_feat)
    print(f"{GREEN}Train, Test and Validation of WineQT Normalized and Standardized successfully!{RESET}\n")

    X_train, X_test, X_val = standardised_train_data, standardised_test_data, standardised_val_data

    # Converting y to 1-hot encoding of 10 classes
    y_train = one_hot_encode_single_class(y_train, 10)
    y_test = one_hot_encode_single_class(y_test, 10)
    y_val = one_hot_encode_single_class(y_val, 10)

    # =================================================================
    #               2.3 Hyperparameter Tuning
    # =================================================================
    best_model_params, loss_dict = hyperparam_tuning("single_class_classifier", 11, 10, X_train, y_train, X_val, y_val, X_test, y_test)
    plot_effect_of_tuning("single_class_classifier", best_model_params, loss_dict)

def MLPMultiClassClassification():
    # Reading the dataset
    data_path = "./data/external/advertisement.csv"
    df = pd.read_csv(data_path)

    # Handle missing values by filling in mean values (No missing values found in advertisement dataset)
    # df.fillna(df.mean(), inplace=True)

    headers = df.columns.to_numpy()
    data = df.to_numpy()

    # Splitting into train, test and val data
    train_data, test_data, val_data = split_data(data, train_percent=TRAIN_PERCENT, test_percent=TEST_PERCENT, val_percent=VAL_PERCENT)

    # Splitting the data into features and labels
    train_features, train_labels = train_data[:, :-1], train_data[:, -1]
    test_features, test_labels = test_data[:, :-1], test_data[:, -1]
    val_features, val_labels = val_data[:, :-1], val_data[:, -1]

    # Each label is a space separated list of multiple lables, so we need to split them
    train_labels = [x.strip().split(" ") for x in train_labels]
    test_labels = [x.strip().split(" ") for x in test_labels]
    val_labels = [x.strip().split(" ") for x in val_labels]

    # One hot encoding the output labels
    one_hot_train_labels = one_hot_encode_multi_class(train_labels)
    one_hot_test_labels = one_hot_encode_multi_class(test_labels)
    one_hot_val_labels = one_hot_encode_multi_class(val_labels)

    # Label encoding all the categorical input features
    list_of_categorical_features = ["gender", "education", "married", "city", "occupation", "most bought item"]
    for categorical_feature in list_of_categorical_features:
        feature_idx = np.where(headers == categorical_feature)[0][0] # Index of the feature to be label encoded
        train_features[:, feature_idx] = label_encode(train_features[:, feature_idx])
        test_features[:, feature_idx] = label_encode(test_features[:, feature_idx])
        val_features[:, feature_idx] = label_encode(val_features[:, feature_idx])
    
    # Normalising and standardising the data
    normalised_train_features, standardised_train_features = normalise_and_standardise(train_features)
    normalised_test_features, standardised_test_features = normalise_and_standardise(test_features)
    normalised_val_features, standardised_val_features = normalise_and_standardise(val_features)

    X_train, y_train = standardised_train_features, one_hot_train_labels
    X_test, y_test = standardised_test_features, one_hot_test_labels
    X_val, y_val = standardised_val_features, one_hot_val_labels

    best_model_params, loss_dict = hyperparam_tuning("multi_class_classifier", 10, 8, X_train, y_train, X_val, y_val, X_test, y_test)
    plot_effect_of_tuning("multi_class_classifier", best_model_params, loss_dict)

def MLPRegression():
    # =================================================================
    #               3.1 Dataset Analysis and Preprocessing 
    # =================================================================
    # Reading the dataset
    data_path = "./data/external/HousingData.csv"
    df = pd.read_csv(data_path)

    # Handle missing values by filling in mean values
    df.fillna(df.mean(), inplace=True)

    headers = df.columns.to_numpy()
    data = df.to_numpy()

    # Describe the dataset using mean, standard deviation, min and max values of each feature
    describe_dataset(data, headers)

    # Plotting histograms of data features
    plot_feature_distribution_histograms(data, headers, save_as="HousingData", n_rows=4, n_cols=4, n_range=14)
    print(f"{GREEN}Histograms of data features saved successfully for HousingData Dataset!{RESET}\n")

    # Splitting the data into train, test and validation sets
    train_data, test_data, val_data = split_data(data, train_percent=TRAIN_PERCENT, test_percent=TEST_PERCENT, val_percent=VAL_PERCENT)
    print(f"{GREEN}Partitioned the housing dataset in {TRAIN_PERCENT}:{TEST_PERCENT}:{VAL_PERCENT}!{RESET}\n")

    # Normalizing and Standardizing the data
    normalised_train_data, standardised_train_data = normalise_and_standardise(train_data)
    normalised_test_data, standardised_test_data = normalise_and_standardise(test_data)
    normalised_val_data, standardised_val_data = normalise_and_standardise(val_data)
    print(f"{GREEN}Train, Test and Validation of Housing Data Normalised and Standardised successfully!{RESET}\n")

    # =================================================================
    #               3.2 MLP Regressor from Scratch
    # =================================================================
    # Splitting the data into features and labels
    X_train, y_train = standardised_train_data[:, :-1], standardised_train_data[:, -1].reshape(-1, 1)
    X_test, y_test = standardised_test_data[:, :-1], standardised_test_data[:, -1].reshape(-1, 1)
    X_val, y_val = standardised_val_data[:, :-1], standardised_val_data[:, -1].reshape(-1, 1)

    hyperparam_tuning("regressor", 13, 1, X_train, y_train, X_val, y_val, X_test, y_test)

    print(f"{MAGENTA}Evaluating the model:{RESET}")
    best_model = MLP(n_op=1)
    best_model.load_model("./best_model_regressor.txt")
    test_MSE = PerformanceMetrics().MSE(y_test, best_model.predict(X_test))
    test_MAE = PerformanceMetrics().MAE(y_test, best_model.predict(X_test))
    print(f"Test Mean Squared Error: {GREEN}{test_MSE}{RESET}")
    print(f"Test Mean Absolute Error: {GREEN}{test_MAE}{RESET}")

    # =================================================================
    #               3.5 MSE vs BCE
    # =================================================================
    # Reading the dataset
    data_path = "./data/external/diabetes.csv"
    df = pd.read_csv(data_path)

    # Handle missing values by filling in mean values
    df.fillna(df.mean(), inplace=True)

    headers = df.columns.to_numpy()
    data = df.to_numpy()

    data = np.float64(data)

    train_data, test_data, val_data = split_data(data, train_percent=TRAIN_PERCENT, test_percent=TEST_PERCENT, val_percent=VAL_PERCENT)
    print(f"{GREEN}Partitioned the diabetes dataset in {TRAIN_PERCENT}:{TEST_PERCENT}:{VAL_PERCENT}!{RESET}\n")

    train_features, train_labels = train_data[:, :-1], train_data[:, -1].reshape(-1, 1)
    test_features, test_labels = test_data[:, :-1], test_data[:, -1].reshape(-1, 1)
    val_features, val_labels = val_data[:, :-1], val_data[:, -1].reshape(-1, 1)

    normalised_train_features, standardised_train_features = normalise_and_standardise(train_features)
    normalised_test_features, standardised_test_features = normalise_and_standardise(test_features)
    normalised_val_features, standardised_val_features = normalise_and_standardise(val_features)

    X_train, y_train = standardised_train_features, train_labels
    X_test, y_test = standardised_test_features, test_labels
    X_val, y_val = standardised_val_features, val_labels

    layers = [8, 1]
    epochs = 100
    lr = 0.01
    batch_size = 32
    act_func = "sigmoid"
    optimiser = "mini_batch"

    train_loss_mse = list()
    val_loss_mse = list()
    train_loss_bce = list()
    val_loss_bce = list()

    print(f"{MAGENTA}Model Hyperparameters are as follows:{RESET}")
    print(f"\tLearning Rate: {GREEN}{lr}{RESET}")
    print(f"\tActivation Function: {GREEN}{act_func}{RESET}")
    print(f"\tOptimiser: {GREEN}{optimiser}{RESET}")
    print(f"\tEpochs: {GREEN}{epochs}{RESET}")
    print(f"\tBatch Size: {GREEN}{batch_size}{RESET}")
    print(f"\tLayers: {GREEN}{layers}{RESET}")
    print()

    logistic_reg_model = MLP(n_ip=layers[0], neurons_per_hidden_layer=[], n_op=layers[-1], learning_rate=lr, activation_func=act_func, optimiser=optimiser, epochs=epochs, batch_size=batch_size, logistic_reg=True)
    start_time = time.time()
    metrics_arr = logistic_reg_model.fit(X_train, y_train, X_val, y_val)
    end_time = time.time()
    print(f"{GREEN}Model fitted in time {end_time - start_time} s!{RESET}\n")

    for metric in metrics_arr:
        train_loss_mse.append(metric["mse_train"])
        val_loss_mse.append(metric["mse_val"])
        train_loss_bce.append(metric["cross_entropy_train"])
        val_loss_bce.append(metric["cross_entropy_val"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].plot(np.arange(epochs), train_loss_bce, label="Train set BCE loss")
    axes[0].plot(np.arange(epochs), val_loss_bce, label="Val set BCE loss")
    axes[0].set_title("Cross Entropy Loss vs Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()

    axes[1].plot(np.arange(epochs), train_loss_mse, label="Train set MSE loss")
    axes[1].plot(np.arange(epochs), val_loss_mse, label="Val set MSE loss")
    axes[1].set_title("Mean Squared Error vs Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./assignments/3/figures/mse_vs_bce.png")
    print(f"{GREEN}MSE and BCE plot for diabetes data saved successfully!{RESET}")

def SpotifyDataset():
    # =================================================================
    #               4.1 Dataset Analysis and Preprocessing 
    # =================================================================
    spotify_dataset = pd.read_csv('./data/external/spotify.csv', quotechar='"')
    spotify_dataset = preprocess(spotify_dataset)

    data = spotify_dataset.to_numpy()

    features = data[:, :-1]
    features = np.float64(features)

    train_data, test_data, val_data = split_data(data, TRAIN_PERCENT, TEST_PERCENT, VAL_PERCENT, shuffle=True)
    train_features, train_labels = train_data[:, :-1], train_data[:, -1].reshape(-1, 1)
    test_features, test_labels = test_data[:, :-1], test_data[:, -1].reshape(-1, 1)
    val_features, val_labels = val_data[:, :-1], val_data[:, -1].reshape(-1, 1)

    train_features = np.float64(train_features)
    test_features = np.float64(test_features)
    val_features = np.float64(val_features)

    model = AutoEncoder(n_ip=14, neurons_per_hidden_layer=[32, 16, 4, 16, 32], n_op=14, learning_rate=0.01, activation_func="sigmoid", optimiser="batch", epochs=100, batch_size=32)

    start_time = time.time()
    model.fit(features)
    model.save_model("./autoencoder_weights_spotify.txt")
    end_time = time.time()
    print(f"{GREEN}Model fitted in time {end_time - start_time} s!{RESET}\n")

    reconstruction_error = model.compute_loss(features, model.predict(features))
    print(f"{GREEN}Reconstruction Error: {reconstruction_error}{RESET}\n")

    reduced_train_features = model.get_latent(train_features)
    reduced_test_features = model.get_latent(test_features)
    reduced_val_features = model.get_latent(val_features)
    print(f"{GREEN}Reduced the data to 4 dimensions using AutoEncoder!{RESET}\n")

    reduced_train_data_with_labels = np.hstack((reduced_train_features, train_labels))
    reduced_test_data_with_labels = np.hstack((reduced_test_features, test_labels))
    reduced_val_data_with_labels = np.hstack((reduced_val_features, val_labels))

    new_header = list()
    n_features = reduced_train_features.shape[1]
    for i in range(n_features):
        new_header.append(f"Feature {i+1}")
    new_header.append("track_genre")

    print(f"{GREEN}Running KNN model on the reduced dataset{RESET}")

    # Initial model's hyperparameters
    distance_metric = "euclidean"
    k = 50

    knn_model = KNN(k, distance_metric)
    knn_model.load_train_test_val_data(new_header, reduced_train_data_with_labels, reduced_test_data_with_labels, reduced_val_data_with_labels)
    knn_model.set_predict_var("track_genre")
    knn_model.use_for_prediction(new_header[:-1])
    start_time = time.time()
    knn_model.fit()
    knn_model.predict("validation")
    metrics = knn_model.get_metrics()
    end_time = time.time()
    time_diff = end_time - start_time

    # Printing metrics
    print(f"""
    k = {k}, Distance Metric = {distance_metric}
    Validation Metrics
                Accuracy:        {round(metrics['accuracy'] * 100, 3)}%\n
                Precision
                        Macro:  {metrics['macro_precision']}    
                        Micro:  {metrics['micro_precision']}\n
                Recall 
                        Macro:   {metrics['macro_recall']}
                        Micro:   {metrics['micro_recall']}\n
                F1 Score
                        Macro:   {metrics['macro_f1_score']}
                        Micro:   {metrics['micro_f1_score']}\n
    Time taken: {round(time_diff, 4)} seconds
    ---------------------------------------------------------------------\n""")

    # =================================================================
    #               4.4 Training MLP classifier on Spotify dataset
    # =================================================================
    print(f"{GREEN}Single Class Classification on Spotify Dataset{RESET}\n")
    # Conveting all the labels into one-hot encoding
    unique_labels = list()
    for row in train_labels:
        for label in row:
            if label not in unique_labels:
                unique_labels.append(label)
    for row in test_labels:
        for label in row:
            if label not in unique_labels:
                unique_labels.append(label)
    for row in val_labels:
        for label in row:
            if label not in unique_labels:
                unique_labels.append(label)

    unique_labels.sort()
    n_classes = len(unique_labels)
    one_hot_train_labels = np.zeros((len(train_labels), n_classes))
    one_hot_test_labels = np.zeros((len(test_labels), n_classes))
    one_hot_val_labels = np.zeros((len(val_labels), n_classes))

    for i in range(len(train_labels)):
        for label in train_labels[i]:
            one_hot_train_labels[i][unique_labels.index(label)] = 1
    for i in range(len(test_labels)):
        for label in test_labels[i]:
            one_hot_test_labels[i][unique_labels.index(label)] = 1
    for i in range(len(val_labels)):
        for label in val_labels[i]:
            one_hot_val_labels[i][unique_labels.index(label)] = 1

    # Custom hyperparameters
    lr = 0.01
    act_func = "sigmoid"
    optimiser = None
    epochs = 100
    batch_size = None
    layers = [14, 32, 64, 128, n_classes]

    # Defining and training model
    model = MLP(n_ip=layers[0], neurons_per_hidden_layer=layers[1:-1], n_op=layers[-1], learning_rate=lr, activation_func=act_func, optimiser=optimiser, epochs=epochs, batch_size=batch_size, loss="cross_entropy")
    start_time = time.time()
    model.fit(train_features, one_hot_train_labels, val_features, one_hot_val_labels)
    end_time = time.time()
    print(f"{GREEN}Model fitted in time {end_time - start_time} s!{RESET}")

    # Calculating precision, recall and f1 scores for each of the classes
    test_p, test_r, test_f1 = PerformanceMetrics().get_precision_recall_f1_onehot_single_class(one_hot_test_labels, model.predict(test_features))

    # Calculating and printing the accuracy, macro precision, macro recall and macro f1 score
    print_aprf1("single_class_classifier", lr, act_func, optimiser, epochs, batch_size, layers, model, test_features, one_hot_test_labels, test_p, test_r, test_f1)

if __name__ == "__main__":
    if USE_WANDB:
        wandb.login()
    MLPSingleClassClassification() 
    MLPMultiClassClassification()
    MLPRegression()
    SpotifyDataset()
