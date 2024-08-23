import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

'''
    ChatGPT Prompt: How to import a module from a different directory
'''
# ====================================================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# ====================================================================================

'''
    Since we have changed the base path, make sure to provide path names from the root of the project
'''

from models.linear_regression.linear_regression import Regression
from models.knn.knn import KNN

# Returns the train, test, val split (in that order)
def split_data(data, train_percent, test_percent, val_percent=None, shuffle=True):
    if train_percent + test_percent > 100:
        raise ValueError("Train and Test percentages should not sum to more than 100")
    if val_percent is None:
        val_percent = 100 - train_percent - test_percent
    else:
        if train_percent + test_percent + val_percent > 100:
            raise ValueError("Train, Test and Validation percentages should not sum to more than 100")
    if shuffle:
        np.random.shuffle(data)
    n_train = int(train_percent * len(data) / 100)
    n_test = int(test_percent * len(data) / 100)
    n_val = int(val_percent * len(data) / 100)

    train_data = data[:n_train]
    test_data = data[n_train:n_train + n_test]
    val_data = data[n_train + n_test:]
    return train_data, test_data, val_data

def save_best_params(path, params):
    with open(path, 'w') as f:
        final_string = ""
        for param in params:
            final_string += f"{param},"
        final_string = final_string[:-1]
        f.write(final_string)

# Visualisation of the data split
def visualise_split(train_data, test_data, validation_data, path_to_save=None):
        x_train = train_data[:,0]
        y_train = train_data[:,1]
        x_validation = validation_data[:,0]
        y_validation = validation_data[:,1]
        x_test = test_data[:,0]
        y_test = test_data[:,1]

        # Plotting the scatter plot of train, validation and test data split
        plt.scatter(x_train, y_train, color='red', label='Train', s=10)
        plt.scatter(x_validation, y_validation, color='blue', label='Validation', s=10)
        plt.scatter(x_test, y_test, color='green', label='Test', s=10)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatter plot of train, validation and test split")
        plt.legend()
        plt.savefig(path_to_save, bbox_inches='tight')
        # plt.show()

# Preprocess the data
def preprocess(data, preprocessed_data_file_path):
    # Renaming the first unnamed column as index
    data = data.rename(columns={'Unnamed: 0': 'index'})

    # Remove exact duplicates - if track id is same, the song is exactly the same
    data = data.drop_duplicates("track_id")

    # Remove duplicate songs in multiple albums
    data = data.drop_duplicates("track_name")

    # Removing unnecessary columns
    unnecessary_cols = ["index", "track_id", "album_name"]
    data = data.drop(columns=unnecessary_cols)

    # Z-score normalization
    cols_to_normalise = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    for col in cols_to_normalise:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std

    # Storing the preprocessed data for future use
    data.to_csv(preprocessed_data_file_path, index=False)
    return data

# Plotting the train and test metrics (MSE, SD and Variance)
def plot_train_test_metrics(x, y, type="scatter", regularisation_type=None):
    fig, axes = plt.subplots(1, len(y), figsize=(10, 5))
    idx = 0
    max_y = 0
    for key, value in y.items():
        max_mse = np.max(value[:, 0])
        max_var = np.max(value[:, 1])
        max_sd = np.max(value[:, 2])
        max_y = max(max_y, max_mse, max_var, max_sd)
        if type == "scatter":
            axes[idx].scatter(x, value[:, 0], color='red', label='MSE', s=10)
            axes[idx].scatter(x, value[:, 1], color='blue', label='Variance', s=10)
            axes[idx].scatter(x, value[:, 2], color="green", label='Std Dev', s=10)
        elif type == "line":
            axes[idx].plot(x, value[:, 0], color='red', label='MSE', linewidth=2)
            axes[idx].plot(x, value[:, 1], color='blue', label='Variance', linewidth=2)
            axes[idx].plot(x, value[:, 2], color="green", label='Std Dev', linewidth=2)
        axes[idx].set_xlabel("Degree")
        axes[idx].set_ylabel("Value")
        axes[idx].set_title(f"MSE, SD and Var Plot for {key.capitalize()} Set")
        axes[idx].legend()
        idx += 1

    # Setting the y-axis limits to be the same for all plots for better comparison (same range)
    if type == "line":
        for i in range(idx):
            axes[i].set_ylim(0, max_y)
    plt.tight_layout()

    # Saving the plot at different locations based on the type of regularisation
    if regularisation_type is None:
        plt.savefig(f"./assignments/1/figures/3.1/final_metrics/linreg_train_test_metrics_{type}_{regularisation_type}.png")
    else:
        plt.savefig(f"./assignments/1/figures/3.2/final_metrics/linreg_train_test_metrics_{type}_{regularisation_type}.png")
    # plt.show()

# Data analysis using various plots
def analyse_data(data):
    # Plotting 1D graph of all the columns to see the distribution (find out outliers)
    cols_to_plot = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    # 11 plots of this and last plot of distribution of genres
    fig, axes = plt.subplots(4, 3, figsize=(8, 8))
    for col in cols_to_plot:
        col_data = data[col].to_numpy()
        row_idx = cols_to_plot.index(col) // 3
        col_idx = cols_to_plot.index(col) % 3
        axes[row_idx, col_idx].scatter(col_data, np.zeros(len(col_data)), s=1)
        axes[row_idx, col_idx].set_title(f"{col.capitalize()} Distribution")
    
    plt.tight_layout()
    plt.savefig("./assignments/1/figures/data_analysis/feature_distribution_after_renormalisation.png")
    plt.show()

    fig, axis = plt.subplots(1, 1, figsize=(15, 8))
    # Plotting the distribution of genres
    genre_counts = data['track_genre'].to_numpy()
    unique_genres, counts = np.unique(genre_counts, return_counts=True)
    '''
        Sorting the genres based on the counts code using ChatGPT
        Prompt: unique_genres, counts = np.unique(genre_counts, return_counts=True) 
                How to sort this based on counts?
    '''
    # =============================================================================
    sorted_indices = np.argsort(counts)
    unique_genres_sorted = unique_genres[sorted_indices]
    counts_sorted = counts[sorted_indices]
    # =============================================================================

    axis.bar(unique_genres_sorted, counts_sorted, color='blue')
    axis.set_xticks(unique_genres_sorted)
    axis.set_xticklabels(unique_genres_sorted, rotation=90)
    axis.set_title("Distribution of Genres")
    plt.tight_layout()
    plt.savefig("./assignments/1/figures/data_analysis/genre_distribution_after_removing_outliers.png")
    plt.show()

    # Set up the visual style
    sns.set_theme(style="whitegrid")

    data_cols = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    # Determine the number of subplots needed
    num_columns = len(data_cols)
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(15, 8))

    # Create a box plot for each numerical column
    for i, column in enumerate(data_cols):
        sns.boxplot(y=data[column], ax=axes[i])
        axes[i].set_title(column)
        axes[i].set_ylabel('')
    plt.tight_layout()
    plt.savefig("./assignments/1/figures/data_analysis/box_plot_each_feature.png")
    plt.show()

    # 1. Distribution of Track Popularity
    plt.figure(figsize=(10, 6))
    sns.histplot(data['popularity'], bins=20, kde=True)
    plt.title('Distribution of Track Popularity')
    plt.xlabel('Popularity')
    plt.ylabel('Frequency')
    plt.savefig("./assignments/1/figures/data_analysis/track_popularity_distribution.png")
    plt.show()

    # 2. Relationship Between Track Attributes
    sns.pairplot(data[['danceability', 'energy', 'loudness', 'valence', 'popularity']], diag_kind='kde')
    plt.suptitle('Pair Plot of Track Attributes', y=1.02)
    plt.savefig("./assignments/1/figures/data_analysis/pair_plot_track_attributes.png")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(data[['danceability', 'energy', 'loudness', 'valence', 'popularity', 'tempo']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Track Attributes')
    plt.savefig("./assignments/1/figures/data_analysis/correlation_heatmap.png")
    plt.show()

    # 3. Genre-wise Analysis
    plt.figure(figsize=(14, 8))
    sns.barplot(x='track_genre', y='popularity', data=data, ci=None)
    plt.title('Average Popularity by Genre')
    plt.xticks(rotation=90)
    plt.savefig("./assignments/1/figures/data_analysis/genre_wise_avg_popularity.png")
    plt.show()

    # 5. Duration vs. Popularity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration_ms', y='popularity', data=data)
    plt.title('Duration vs. Popularity')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Popularity')
    plt.savefig("./assignments/1/figures/data_analysis/duration_vs_popularity.png")
    plt.show()

    # 6. Time Signature
    plt.figure(figsize=(8, 6))
    sns.countplot(x='time_signature', data=data)
    plt.title('Distribution of Time Signature')
    plt.xlabel('Time Signature')
    plt.show()

    # 7. Artists’ Contribution
    top_artists = data['artists'].value_counts().head(10).index
    filtered_data = data[data['artists'].isin(top_artists)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # First subplot: Number of Tracks per Top 10 Artists
    sns.countplot(y='artists', data=filtered_data, order=top_artists, ax=axes[0])
    axes[0].set_title('Number of Tracks per Top 10 Artists')

    # Second subplot: Total Popularity by Top 10 Artists
    sns.barplot(y='artists', x='popularity', data=filtered_data, ci=None, order=top_artists, ax=axes[1])
    axes[1].set_title('Total Popularity by Top 10 Artists')
    plt.tight_layout()
    plt.savefig("./assignments/1/figures/data_analysis/artists_contribution.png")
    plt.show()

# KNN Model
def knn(headers, data):
    k = 10
    dist_metric = "euclidean"
    knn_model = KNN(k, dist_metric)
    train_data, test_data, val_data = split_data(data, 80, 10, 10, shuffle=True)
    knn_model.load_train_data(headers, train_data)
    knn_model.set_predict_var("track_genre")
    knn_model.use_for_prediction(["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"])
    predictions = knn_model.predict(test_data)

    # Code just for testing and debuggin purpose
    ok = 0
    false = 0
    for i in range(len(predictions)):
        if predictions[i] == test_data[i][-1]:
            ok += 1
        else:
            false += 1
    print(ok, false)

# Linear Regression Model
def linear_regression(data, max_k=1, regularisation_method=None, lamda=0):
    # Hyperparameters
    lr = 0.1                                # Learning Rate
    diff_threshold = 0.0003                 # Difference Threshold, to check if the model has converged
    seed = 10                               # Seed for reproducibility
    max_epochs_after_which_to_stop = 100    # Maximum number of epochs after which to stop training

    # Setting up the model based on the regularisation method
    if regularisation_method == "l1":
        linreg = Regression(regularization_method="l1", lamda=lamda)
    elif regularisation_method == "l2":
        linreg = Regression(regularization_method="l2", lamda=lamda)
    else:
        linreg = Regression()

    '''
        If you wanna load the model from a file, uncomment the below line
    '''
    # =============================================================================
    # linreg.load_model(f"./assignments/1/best_model_params_{regularisation_method}.txt")
    # =============================================================================

    # Note that to better see the results of regularisation and overfitting try reducing the training sample size to about 4-5%
    train_data, test_data, val_data = split_data(data, 80, 10, 10, shuffle=True)
    linreg.load_train_test_validation_data(train_data, test_data, val_data)

    # Save at different paths based on the type of regularisation
    if regularisation_method is None:
        visualise_split(train_data, test_data, val_data, path_to_save=f"./assignments/1/figures/3.1/data_split/train_test_validation_split_{regularisation_method}_linreg.png")
    else:
        visualise_split(train_data, test_data, val_data, path_to_save=f"./assignments/1/figures/3.2/data_split/train_test_validation_split_{regularisation_method}_linreg.png")
    
    # Arrays to store metric values for each degree of polynomial
    train_metrics = np.zeros((max_k, 3))
    test_metrics = np.zeros((max_k, 3))

    min_test_err = np.inf
    best_params = None

    # Train model for different degrees of polynomial
    for k in range(1, max_k + 1):
        # Saving the images only when animation is required to be created i.e. when regularisation is not applied and for k = 1, 5, 10, 15, 20
        # Epoch images need to be saved if you want to animate at the end
        save_epoch_imgs = False
        if k in [1, 5, 10, 15, 20] and regularisation_method is None:
            save_epoch_imgs = True
        converged_epoch = linreg.fit(k, lr, diff_threshold, save_epoch_imgs=save_epoch_imgs, seed=seed, max_epochs=max_epochs_after_which_to_stop)
        
        # Train Metrics
        mse_train, var_train, sd_train = linreg.get_metrics("train")
        train_metrics[k - 1] = np.array([mse_train, var_train, sd_train])
        
        # Test Metrics
        mse_test, var_test, sd_test = linreg.get_metrics("test")
        test_metrics[k - 1] = np.array([mse_test, var_test, sd_test])

        # Printing information about the model with degree k
        print(f"""Degree: {k}\n
                  Method: {regularisation_method}\n
                  Seed: {seed}\n
                  Converged at epoch: {converged_epoch}\n
                  Final Parameters: {linreg.get_params()}\n
                  Training set: MSE: {mse_train}, Variance: {var_train}, Standard Deviation {sd_train}\n
                  Test set:     MSE: {mse_test}, Variance: {var_test}, Standard Deviation: {sd_test}\n
-----------------------------------------------------------------------------------------------------------------""")
        if mse_test < min_test_err:
            min_test_err = mse_test
            best_params = linreg.get_params()

        # Creating the animations
        if k in [1, 5, 10, 15, 20] and regularisation_method is None:
            linreg.animate_training(f"./assignments/1/figures/3.1/gif/lin_reg_{regularisation_method}_regu_animation_{k}.gif")
        
        # Saving the final output figure for each k
        if regularisation_method is None:
            linreg.visualise_fit(train_data, "save", output_path=f"./assignments/1/figures/3.1/final_regression_curve/linreg_{regularisation_method}_regu_{k}.png")
        else:
            linreg.visualise_fit(train_data, "save", output_path=f"./assignments/1/figures/3.2/final_regression_curve/linreg_{regularisation_method}_regu_{k}.png")
    
    # Plotting the graph for train and test metrics
    if k == 1:
        plot_train_test_metrics(x=np.arange(1, max_k+1), 
                            y={"train": train_metrics, "test": test_metrics}, 
                            type="scatter",
                            regularisation_type=regularisation_method)
    else:
        plot_train_test_metrics(x=np.arange(1, max_k+1), 
                            y={"train": train_metrics, "test": test_metrics}, 
                            type="line",
                            regularisation_type=regularisation_method)
    
    '''
        If you wanna save the best model, uncomment the below
    '''
    # =============================================================================
    # save_best_params(best_params, f"./assignments/1/best_model_params_{regularisation_method}.txt")
    # =============================================================================

if __name__ == "__main__":
    '''
        Data Preprocessing and reading Part
    '''
    # knn_data_file_path = './data/external/spotify.csv' 
    # knn_preprocessed_data_file_path = './data/interim/1/preprocessed_spotify.csv'
    # # Some data points have commas in them, so we need to use quotechar to read the file
    # # Source: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    # try:
    #     # Checking if preprocessed data is already present
    #     knn_data = pd.read_csv(knn_preprocessed_data_file_path, quotechar='"')
    # except FileNotFoundError:
    #     knn_data = pd.read_csv(knn_data_file_path, quotechar='"')
    #     knn_data = preprocess(knn_data, knn_preprocessed_data_file_path)

    '''
        Data Analysis Part
    '''
    # analyse_data(knn_data)

    '''
        KNN Part
    '''
    # headers = list(knn_data.columns)
    # knn_data = knn_data.to_numpy()
    # knn(headers, knn_data)

    '''
        Linear Regression Part
    '''
    # # Degree 1
    # linreg_data_path = "./data/external/linreg.csv"
    # linreg_data = np.genfromtxt(linreg_data_path, delimiter=',', skip_header=True)
    # linear_regression(linreg_data)

    # Degree > 1
    linreg_data_path = "./data/external/linreg.csv"
    linreg_data = np.genfromtxt(linreg_data_path, delimiter=',', skip_header=True)
    linear_regression(linreg_data, max_k=20)

    # L1 & L2 regularisation
    regularisation_data_path = "./data/external/regularisation.csv"
    regularisation_data = np.genfromtxt(regularisation_data_path, delimiter=',', skip_header=True)
    '''
        For comparison we can do linear regression for the regularisation.csv data without regularisation
        For better visualisation of regularisation taking place, try to keep the training set size about 4-5%
    '''
    # =============================================================================
    # linear_regression(regularisation_data, 20)
    # =============================================================================

    linear_regression(regularisation_data, 20, "l1", 10)
    linear_regression(regularisation_data, 20, "l2", 10)