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
wandb.login()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.MLP.MLP import MLP_Classifier, MLP_Regressor
from models.AutoEncoders.AutoEncoders import AutoEncoder
from performance_measures.performance_measures import PerformanceMetrics

TRAIN_PERCENT = 70
TEST_PERCENT = 15
VAL_PERCENT = 15

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

def describe_dataset(data):
    mean, sd, min_vals, max_vals = np.mean(data, axis=0), np.std(data, axis=0), np.min(data, axis=0), np.max(data, axis=0)
    print(f"{GREEN}Mean, Standard Deviation, Min, Max values of the dataset features:{RESET}")
    attr_table = pt.PrettyTable()
    attr_table.field_names = ["Attribute", "Mean", "Standard Deviation", "Min", "Max"]
    for i in range(len(headers)):
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
    cols_to_normalise = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    for col in cols_to_normalise:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std

    return data

def convert_to_one_hot_encoding(data, n_classes):
    one_hot_data = np.zeros((data.shape[0], n_classes))
    for i in range(data.shape[0]):
        one_hot_data[i][int(data[i]) - 1] = 1
    return one_hot_data

def print_aprf1(lr, act_func, optimiser, epochs, batch_size, layers, classifier_model, X_test, y_test, test_p, test_r, test_f1):
    accuracy = round(classifier_model.compute_acc(X_test, y_test) * 100, 3)
    macro_precision = round(np.mean(test_p) * 100, 3)
    macro_recall = round(np.mean(test_r) * 100, 3)
    macro_f1 = round(np.mean(test_f1) * 100, 3)
    
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

        Macro Test F1 Score: {GREEN}{macro_f1}%{RESET}

        -----------------------------------------------------------

        Micro Test Precision: {GREEN}{[float(x) for x in test_p]}{RESET}

        Micro Test Recall: {GREEN}{[float(x) for x in test_r]}{RESET}

        Micro Test F1 Score: {GREEN}{[float(x) for x in test_f1]}{RESET}
    """)
    return [accuracy, macro_precision, macro_recall, macro_f1]

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
    # describe_dataset(data)

    # plot_feature_distribution_histograms(data, headers, save_as="WineQT")
    # print(f"{GREEN}Histograms of data features saved successfully for WineQT Dataset!{RESET}\n")

    train_data, test_data, val_data = split_data(data, train_percent=TRAIN_PERCENT, test_percent=TEST_PERCENT, val_percent=VAL_PERCENT)
    print(f"{GREEN}Partitioned the WineQT dataset in {TRAIN_PERCENT}:{TEST_PERCENT}:{VAL_PERCENT}!{RESET}\n")
    
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
    y_train = convert_to_one_hot_encoding(y_train, 10)
    y_test = convert_to_one_hot_encoding(y_test, 10)
    y_val = convert_to_one_hot_encoding(y_val, 10)

    # =================================================================
    #               2.3 Hyperparameter Tuning
    # =================================================================
    learning_rates = [0.001, 0.01, 0.1]
    activation_funcs = ["sigmoid", "relu", "tanh", "linear"]
    optimisers = ["batch", "sgd", "mini_batch", None]
    epochs = 100
    batch_size = 20
    layers_arr = [[11, 15, 15, 10],
                  [11, 20, 20, 10],
                  [11, 15, 15, 15, 10],
                  [11, 256, 256, 10],
                  [11, 256, 64, 32, 10]]

    for optimiser in optimisers:
        for layers in layers_arr:
            for lr in learning_rates:
                for act_func in activation_funcs:

                    hyperparameters = {
                        "learning_rate": lr,
                        "activation_func": act_func,
                        "optimiser": optimiser,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "layers": layers
                    }

                    wandb.init(
                        project="assignment3-MLPClassifier_SingleClass",
                        name=f"{optimiser}_{lr}_{act_func}_layeridx{layers_arr.index(layers)}",
                        config=hyperparameters
                    )

                    classifier_model = MLP_Classifier(
                                    n_ip=layers[0], 
                                    neurons_per_hidden_layer=layers[1:-1], 
                                    n_op=layers[-1],
                                    learning_rate=lr,
                                    activation_func=act_func,
                                    optimiser=optimiser,
                                    epochs=epochs,
                                    batch_size=batch_size
                                )

                    start_time = time.time()
                    metrics_arr = classifier_model.fit(X_train, y_train, X_val, y_val)
                    classifier_model.save(f"./weights/{optimiser}_{lr}_{act_func}_layeridx{layers_arr.index(layers)}.txt")
                    end_time = time.time()
                    print(f"{GREEN}Model Training completed successfully in time {end_time - start_time} s!{RESET}\n")

                    test_p, test_r, test_f1 = PerformanceMetrics().get_precision_recall_f1_onehot(y_test, classifier_model.predict(X_test))

                    # Temporary delete this writing hyperparams to file part after complete working
                    aprf1 = print_aprf1(lr, act_func, optimiser, epochs, batch_size, layers, classifier_model, X_test, y_test, test_p, test_r, test_f1)
                    hyperparameters["accuracy"] = aprf1[0]
                    hyperparameters["precision"] = aprf1[1]
                    hyperparameters["recall"] = aprf1[2]
                    hyperparameters["f1"] = aprf1[3]

                    with open(f"./params/{optimiser}_{lr}_{act_func}_layeridx{layers_arr.index(layers)}.txt", 'w') as f:
                        for key, value in hyperparameters.items():
                            f.write(f'{key} : {value}\n')
                    
                    for metric in metrics_arr:
                        wandb.log(metric)
                    
                    wandb.finish()
    exit()

def MLPMultiClassClassification():
    pass

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
    describe_dataset(data)

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

    lrs = [0.001, 0.01, 0.1]
    act_funcs = ["sigmoid", "relu", "tanh", "linear"]
    optimisers = ["sgd", "batch", "mini_batch", None]
    epochs = 100
    batch_size = 20
    layers = [13, 10, 10, 1]

    # lr = 0.001
    # act_func = "sigmoid"
    # optimiser = "sgd"
    # epochs = 100
    # batch_size = 20
    # layers = [13, 10, 10, 1]

    for lr in lrs:
        for act_func in act_funcs:
            for optimiser in optimisers:
                wandb.init(
                    project="assignment3-MLPRegressor",
                    config={
                        "learning_rate": lr,
                        "activation_func": act_func,
                        "optimiser": optimiser,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "layers": layers
                    }
                )

                # Training the MLP Regressor model
                regressor_model = MLP_Regressor(
                    n_ip=layers[0], 
                    neurons_per_hidden_layer=layers[1:-1], 
                    n_op=layers[-1],
                    learning_rate=lr,
                    activation_func=act_func,
                    optimiser=optimiser,
                    epochs=epochs,
                    batch_size=batch_size
                )

                start_time = time.time()
                metrics_arr = regressor_model.fit(X_train, y_train, X_val, y_val)
                end_time = time.time()
                print(f"{GREEN}Model Training completed successfully in time {end_time - start_time}!{RESET}\n")

                # Logging the metrics to Weights and Biases
                for metric_dict in metrics_arr:
                    wandb.log(metric_dict)
                print(f"{GREEN}Metrics logged successfully to Weights and Biases!{RESET}\n")

    # wandb.init(
    #     project="assignment3-MLPRegressor",
    #     config={
    #         "learning_rate": lr,
    #         "activation_func": act_func,
    #         "optimiser": optimiser,
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "layers": layers
    #     }
    # )

    # # Training the MLP Regressor model
    # regressor_model = MLP_Regression(
    #     n_ip=layers[0], 
    #     neurons_per_hidden_layer=layers[1:-1], 
    #     n_op=layers[-1],
    #     learning_rate=lr,
    #     activation_func=act_func,
    #     optimiser=optimiser,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )

    # start_time = time.time()
    # metrics_arr = regressor_model.fit(X_train, y_train, X_val, y_val)
    # end_time = time.time()
    # print(f"{GREEN}Model Training completed successfully in time {end_time - start_time}!{RESET}\n")

    # # Logging the metrics to Weights and Biases
    # for metric_dict in metrics_arr:
    #     wandb.log(metric_dict)
    # print(f"{GREEN}Metrics logged successfully to Weights and Biases!{RESET}\n")

def SpotifyDataset():
    # =================================================================
    #               4.1 Dataset Analysis and Preprocessing 
    # =================================================================
    # spotify_dataset = pd.read_csv('./data/external/spotify.csv', quotechar='"')
    # spotify_dataset = preprocess(spotify_dataset)
    # headers = spotify_dataset.columns.to_numpy()
    # data = spotify_dataset.to_numpy()
    pass

if __name__ == "__main__":
    MLPSingleClassClassification()
    MLPMultiClassClassification()
    MLPRegression()
    SpotifyDataset()