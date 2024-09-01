import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns
import time
import itertools

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

'''
    The code for importing file from sibling directories has been referenced using ChatGPT
    ChatGPT Prompt: How to import a module from a sibling directory
'''
# ====================================================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.linear_regression.linear_regression import Regression
from models.knn.knn import KNN
# ====================================================================================

'''
    Since we have changed the base path, make sure to provide path names from the root of the project
'''

# Returns the train, test, val split (in that order)
def split_data(data, train_percent, test_percent, val_percent=None, shuffle=True, seed=42):
    if train_percent + test_percent > 100:
        raise ValueError("Train and Test percentages should not sum to more than 100")
    if val_percent is None:
        val_percent = 100 - train_percent - test_percent
    else:
        if train_percent + test_percent + val_percent > 100:
            raise ValueError("Train, Test and Validation percentages should not sum to more than 100")
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(data)
    n_train = int(train_percent * len(data) / 100)
    n_test = int(test_percent * len(data) / 100)
    n_val = int(val_percent * len(data) / 100)

    train_data = data[:n_train]
    test_data = data[n_train:n_train + n_test]
    val_data = data[n_train + n_test:]
    return train_data, test_data, val_data

# Saves the list of parameters passed to it in a text file in csv format
def save_params(path, params):
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
        plt.close()

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
        plt.savefig(f"./assignments/1/figures/linreg/final_metrics/linreg_train_test_metrics_{type}_{regularisation_type}.png")
    else:
        plt.savefig(f"./assignments/1/figures/linreg_regu/final_metrics/linreg_train_test_metrics_{type}_{regularisation_type}.png")
    # plt.show()

'''
    The following code for generating all possible combinations is given by ChatGPT
    Prompt: Given a list of elements, generate all possible combinations of the elements
'''
# =============================================================================
def generate_all_combinations(lst):
    all_combinations = []
    for r in range(1, len(lst) + 1):
        combinations = list(itertools.combinations(lst, r))
        all_combinations.extend(combinations)
    return all_combinations
# =============================================================================

# Data analysis using various plots
def analyse_data(data):
    numerical_features = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

    # Frequency of each genre in sorted order
    fig, axis = plt.subplots(1, 1, figsize=(15, 8))
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
    plt.savefig("./assignments/1/figures/data_analysis/genre_vs_frequency.png")
    print("Genre vs Frequency plot saved.")
    plt.close()

    # Setting global theme for seaborn
    sns.set_theme(style="whitegrid")

    # Box-plot for each numerical feature
    num_columns = len(numerical_features)
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(15, 8))

    for i, column in enumerate(numerical_features):
        sns.boxplot(y=data[column], ax=axes[i])
        axes[i].set_title(column)
        axes[i].set_ylabel('')
    plt.tight_layout()
    plt.savefig("./assignments/1/figures/data_analysis/box_plot_each_feature.png")
    print("Box plot for each numerical feature saved.")
    plt.close()

    # Popularity distribution of the songs, in which bin does most of the songs fall
    plt.figure(figsize=(10, 6))
    sns.histplot(data['popularity'], bins=20, kde=True)
    plt.title('Distribution of Track Popularity')
    plt.xlabel('Popularity')
    plt.ylabel('Frequency')
    plt.savefig("./assignments/1/figures/data_analysis/track_popularity_distribution.png")
    print("Popularity distribution plot saved.")
    plt.close()

    # Pair-plots between imp attributes
    '''
        Initially I had plot all the features vs all features pairplot, but after analysing that, I understood
        some features are not necessary in the pariplot as they are too skewed or too evenly distributed to give 
        any meaningful information so redefined the pairplot features array
    '''
    pair_plot_features = ["popularity", "danceability", "energy", "loudness", "tempo"]
    sns.pairplot(data[pair_plot_features], diag_kind='kde', plot_kws={'s': 2})
    plt.suptitle('Pair Plot of Track Attributes', y=1.02)
    plt.savefig("./assignments/1/figures/data_analysis/pair_plot_track_attributes.png")
    print("Pair plot of track attributes saved.")
    plt.close()

    # Genre vs average value of each feature
    '''
        The below plot for the distribution of numerical data - genre wise has been generated by ChatGPT
        Prompt: Given data and list of features, generate 12 subplots in 4x3 format showing genre vs. average value
                graph. Since all the subplots will have the same legend (track_genre) thus plot a custom legend on
                the right of the whole plot instead of plotting individually.
    '''
    # ====================================================================================
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Average Features by Genre', fontsize=16)
    axes = axes.flatten()
    
    # Generating color palette
    genres = data['track_genre'].unique()
    color_palette = sns.color_palette("husl", n_colors=len(genres))
    color_dict = dict(zip(genres, color_palette))

    # Plotting subplots
    for i, feature in enumerate(numerical_features):
        sns.barplot(data=data, x='track_genre', y=feature, ax=axes[i], hue='track_genre', errorbar=None, dodge=False)
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel('Genre')

    # Custom legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_dict[genre], edgecolor='none') for genre in genres]
    fig.legend(legend_elements, genres, loc='center left', bbox_to_anchor=(1, 0.5), title='Genre', fontsize='small', title_fontsize='medium')
    plt.tight_layout(rect=[0, 0, 0.9, 1])   # Resizing the layout to fit the legend
    plt.savefig('./assignments/1/figures/data_analysis/genre_wise_avg_popularity.png', bbox_inches='tight')
    print("Genre wise average popularity plot saved.")
    plt.close()
    # ====================================================================================

    # Popularity vs Songs of artist, how many songs did they publish and how popular they are
    top_artists = data['artists'].value_counts().head(10).index
    filtered_data = data[data['artists'].isin(top_artists)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # First subplot: Number of Tracks per Top 10 Artists
    sns.countplot(y='artists', data=filtered_data, order=top_artists, ax=axes[0])
    axes[0].set_title('Number of Tracks per Top 10 Artists')

    # Second subplot: Total Popularity by Top 10 Artists
    sns.barplot(y='artists', x='popularity', data=filtered_data, errorbar=None, order=top_artists, ax=axes[1])
    axes[1].set_title('Total Popularity by Top 10 Artists')
    plt.tight_layout()
    plt.savefig("./assignments/1/figures/data_analysis/artists_contribution.png")
    print("Artists contribution plot saved.")
    plt.close()

    # Distribution of features - to find skewness
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        sns.histplot(data=data, x=feature, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')

    plt.tight_layout()
    plt.savefig('./assignments/1/figures/data_analysis/distribution_of_features.png')
    print("Distribution of features plot saved.")
    plt.close()

    # Heatmap between all possible pairs of numerical features
    correlation_matrix = data[numerical_features].corr()
    plt.figure(figsize=(12, 10))    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('./assignments/1/figures/data_analysis/corelation_heatmap_all_features.png')
    print("Correlation heatmap saved.")
    plt.close()

def read_dataset(data_set):
    if data_set == 1:
        # Dataset 1 - spotify.csv
        knn_data_file_path = './data/external/spotify.csv' 
        knn_preprocessed_data_file_path = './data/interim/1/preprocessed_spotify.csv'
        # Some data points have commas in them, so we need to use quotechar to read the file
        # Source: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
        try:
            # Checking if preprocessed data is already present
            knn_data = pd.read_csv(knn_preprocessed_data_file_path, quotechar='"')
        except FileNotFoundError:
            knn_data = pd.read_csv(knn_data_file_path, quotechar='"')
            knn_data = preprocess(knn_data, knn_preprocessed_data_file_path)
        headers = list(knn_data.columns)
        data = knn_data.to_numpy()
        return headers, data
    elif data_set == 2:
        # Dataset 2 - spotify-2.csv
        train_data_file_path = './data/external/spotify-2/train.csv'
        test_data_file_path = './data/external/spotify-2/test.csv'
        validate_data_file_path = './data/external/spotify-2/validate.csv'

        train_preprocessed_data_file_path = './data/interim/1/preprocessed_spotify2_train.csv'
        test_preprocessed_data_file_path = './data/interim/1/preprocessed_spotify2_test.csv'
        validate_preprocessed_data_file_path = './data/interim/1/preprocessed_spotify2_validate.csv'

        # Some data points have commas in them, so we need to use quotechar to read the file
        # Source: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
        try:
            # Checking if preprocessed data is already present
            train_data2 = pd.read_csv(train_preprocessed_data_file_path, quotechar='"')
            test_data2 = pd.read_csv(test_preprocessed_data_file_path, quotechar='"')
            validate_data2 = pd.read_csv(validate_preprocessed_data_file_path, quotechar='"')
        except FileNotFoundError:
            train_data2 = pd.read_csv(train_data_file_path, quotechar='"')
            test_data2 = pd.read_csv(test_data_file_path, quotechar='"')
            validate_data2 = pd.read_csv(validate_data_file_path, quotechar='"')
            train_data2 = preprocess(train_data2, train_preprocessed_data_file_path)
            test_data2 = preprocess(test_data2, test_preprocessed_data_file_path)
            validate_data2 = preprocess(validate_data2, validate_preprocessed_data_file_path)

        headers = list(train_data2.columns)
        train_data2 = train_data2.to_numpy()
        test_data2 = test_data2.to_numpy()
        validate_data2 = validate_data2.to_numpy()
        return headers, train_data2, test_data2, validate_data2

def sklearn_model(k, train_data_set_size=80):
    data = pd.read_csv('./data/external/spotify.csv')
    data = data.drop(columns=['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'])

    label_encoders = {}
    categorical_columns = ['explicit', 'track_genre']

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop(columns=['track_genre'])
    y = data['track_genre']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=(1-(train_data_set_size/100)), random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    end_time = time.time()
    time_diff = end_time - start_time

    return accuracy, time_diff

# KNN Model
def knn():
    # Initial model's hyperparameters
    initial_k = 10
    initial_distance_metric = "euclidean"

    print("Starting with dataset 1")
    headers, data = read_dataset(1)

    distance_metrics = ["euclidean", "manhattan", "cosine"]
    k_values = [5, 10, 15, 20, 25, 50, 100]

    train_data, test_data, val_data = split_data(data, 80, 10, 10, shuffle=True)

    all_numeric_features = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    combinations_to_use = [
        all_numeric_features,
        ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness'],
        ["popularity", "danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    ]
    # Did not have time to run this code, but this is the code to generate all possible combinations of features
    # combinations_to_use = generate_all_combinations(all_numeric_features)
    # combinations_to_use = [list(combination) for combination in combinations_to_use]

    models = []
    accuracies = []
    times = []
    combinations = []

    for idx, combination in enumerate(combinations_to_use):
        for k in k_values:
            for distance_metric in distance_metrics:
                # k, distance_metric, combination_index
                models.append((k, distance_metric))
                combinations.append(idx)
                knn_model = KNN(k, distance_metric)
                knn_model.load_train_test_val_data(headers, train_data, test_data, val_data)
                knn_model.set_predict_var("track_genre")
                knn_model.use_for_prediction(combination)
                start_time = time.time()
                knn_model.fit()
                knn_model.predict("validation")
                metrics = knn_model.get_metrics()
                end_time = time.time()
                time_diff = end_time - start_time

                # Printing metrics
                print(f"""
                k = {k}, Distance Metric = {distance_metric}
                Validation Metrics for Dataset 1
                Combination index used: {idx}\n
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
                accuracies.append(round(metrics['accuracy'] * 100, 3))
                times.append(time_diff)

    # Finding the top 10 models
    top_10_indices = np.argsort(accuracies)[-10:]
    top_10_models = [models[i] for i in top_10_indices]
    top_10_accuracies = [accuracies[i] for i in top_10_indices]
    top_10_times = [times[i] for i in top_10_indices]
    top_10_combinations = [combinations[i] for i in top_10_indices]

    # Reversing all 3 lists to get the top 10 models in descending order of accuracy
    top_10_models  = top_10_models[::-1]
    top_10_accuracies = top_10_accuracies[::-1]
    top_10_times = top_10_times[::-1]
    top_10_combinations = top_10_combinations[::-1]

    # Best model hyperparameters
    best_k = top_10_models[0][0]
    best_distance_metric = top_10_models[0][1]
    # My optimised model is the same as the best model, so I am setting the optimised model to the best model
    optimised_k = best_k
    optimised_distance_metric = best_distance_metric

    # Printing the top 10 models
    print("\nTop 10 Models\n")
    for i in range(min(10, len(models))):
        print(f"Model: {top_10_models[i]}, Combination: {top_10_combinations[i]}, Accuracy: {top_10_accuracies[i]}, Time: {top_10_times[i]} seconds")

    # Plottig k vs accuracy plot
    distance_metric_of_choice = "euclidean"
    accuracies_to_plot = []
    for k in k_values:
        for idx, model in enumerate(models):
            if model[0] == k and model[1] == distance_metric_of_choice:
                accuracies_to_plot.append(accuracies[idx])
                break

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies_to_plot, marker='o')
    plt.title(f"K vs Accuracy for {distance_metric_of_choice.capitalize()} Distance Metric")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig(f"./assignments/1/figures/knn/k_vs_accuracy{distance_metric_of_choice}.png")
    print("K vs Accuracy plot saved.")
    plt.close()

    print("\nStarting with dataset 2")
    headers2, train_data2, test_data2, validate_data2 = read_dataset(2)

    all_numeric_features2 = all_numeric_features

    knn2 = KNN(best_k, best_distance_metric)
    knn2.load_train_test_val_data(headers2, train_data2, test_data2, validate_data2)
    knn2.set_predict_var("track_genre")
    knn2.use_for_prediction(all_numeric_features2)
    start_time = time.time()
    knn2.fit()
    knn2.predict("validation")
    metrics = knn2.get_metrics()
    end_time = time.time()
    time_diff = end_time - start_time

    # Printing metrics
    print(f"""
    k = {best_k}, Distance Metric = {best_distance_metric}
    Validation Metrics for Dataset 2\n
                Accuracy:        {round(metrics['accuracy'] * 100, 3)}%\n
                Precision
                        Macro:  {metrics['macro_precision']}
                        Micro:  {metrics['micro_precision']}\n
                Recall
                        Macro:  {metrics['macro_recall']}
                        Micro:  {metrics['micro_recall']}\n
                F1 Score
                        Macro:  {metrics['macro_f1_score']}
                        Micro:  {metrics['micro_f1_score']}\n
    Time taken: {round(time_diff, 4)} seconds
    ---------------------------------------------------------------------\n""")

    # Time taken vs Training set size
    initial_time_taken_arr = []
    best_time_taken_arr = []
    optimised_time_taken_arr = []
    sklearn_time_taken_arr = []

    print("Starting with training set size vs time taken\n")
    training_set_sizes = [80, 70, 60, 50, 40, 30, 20, 10] # In percentage
    for training_set_size in training_set_sizes:
        print("Current training set size: ", training_set_size)
        test_size = int((100 - training_set_size) / 2)
        val_size = int((100 - training_set_size) / 2)
        train_data, test_data, val_data = split_data(data, training_set_size, test_size, val_size, shuffle=True)

        # Best Model; Same as optimised model
        knn_model = KNN(best_k, best_distance_metric)
        knn_model.load_train_test_val_data(headers, train_data, test_data, val_data)
        knn_model.set_predict_var("track_genre")
        knn_model.use_for_prediction(all_numeric_features)
        start_time = time.time()
        knn_model.fit()
        knn_model.predict("validation")
        end_time = time.time()
        time_diff = end_time - start_time
        best_time_taken_arr.append(time_diff)
        optimised_time_taken_arr.append(time_diff)
        print("Best Model Time Taken:", time_diff)

        # Initial Model
        knn_model = KNN(initial_k, initial_distance_metric)
        knn_model.load_train_test_val_data(headers, train_data, test_data, val_data)
        knn_model.set_predict_var("track_genre")
        knn_model.use_for_prediction(all_numeric_features)
        start_time = time.time()
        knn_model.fit()
        knn_model.predict("validation")
        end_time = time.time()
        time_diff = end_time - start_time
        initial_time_taken_arr.append(time_diff)
        print("Initial Model Time Taken:", time_diff)

        # Sklearn Model
        accuracy, time_diff = sklearn_model(best_k, training_set_size)
        sklearn_time_taken_arr.append(time_diff)
        print("Sklearn Model Time Taken:", time_diff)

    plt.figure(figsize=(10, 6))
    plt.plot(training_set_sizes, best_time_taken_arr, marker='o', color='red')
    plt.plot(training_set_sizes, optimised_time_taken_arr, marker='o', color='blue')
    plt.plot(training_set_sizes, initial_time_taken_arr, marker='o', color='green')
    plt.plot(training_set_sizes, sklearn_time_taken_arr, marker='o', color='pink')

    plt.title("Training Set Size vs Time Taken")
    plt.xlabel("Training Set Size")
    plt.ylabel("Time Taken (in seconds)")
    plt.savefig("./assignments/1/figures/knn/training_set_size_vs_time.png")
    print("Training set size vs time taken plot saved.")
    plt.close()

# Linear Regression Model
def linear_regression(data, max_k=1, regularisation_method=None, lamda=0):
    # Hyperparameters (lr and diff_threshold are decided after multiple trial and errors)
    lr = 0.1                                # Learning Rate
    diff_threshold = 0.0003                 # Difference Threshold, to check if the model has converged
    seed = 42                               # Seed for reproducibility
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
        visualise_split(train_data, test_data, val_data, path_to_save=f"./assignments/1/figures/linreg/data_split/train_test_validation_split_{regularisation_method}_linreg.png")
    else:
        visualise_split(train_data, test_data, val_data, path_to_save=f"./assignments/1/figures/linreg_regu/data_split/train_test_validation_split_{regularisation_method}_linreg.png")
    
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
            linreg.animate_training(f"./assignments/1/figures/linreg/gif/lin_reg_{regularisation_method}_regu_animation_{k}.gif")

        # Saving the final output figure for each k
        if regularisation_method is None:
            linreg.visualise_fit(train_data, "save", output_path=f"./assignments/1/figures/linreg/final_regression_curve/linreg_{regularisation_method}_regu_{k}.png")
        else:
            linreg.visualise_fit(train_data, "save", output_path=f"./assignments/1/figures/linreg_regu/final_regression_curve/linreg_{regularisation_method}_regu_{k}.png")
    
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
    
    print(f"\nBest Parameters: {best_params}\n")
    '''
        If you wanna save the best model, uncomment the below
    '''
    # =============================================================================
    save_params(f"./assignments/1/best_model_params_{regularisation_method}.txt", best_params.tolist())
    # =============================================================================

if __name__ == "__main__":
    '''
        Data Preprocessing and reading Part
    '''
    data_analysis_file_path = './data/external/spotify.csv' 
    data_analysis_preprocessed_data_file_path = './data/interim/1/preprocessed_spotify.csv'
    # Some data points have commas in them, so we need to use quotechar to read the file
    # Source: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    try:
        # Checking if preprocessed data is already present
        data_for_analysis = pd.read_csv(data_analysis_preprocessed_data_file_path, quotechar='"')
    except FileNotFoundError:
        data_for_analysis = pd.read_csv(data_analysis_file_path, quotechar='"')
        data_for_analysis = preprocess(data_for_analysis, data_analysis_preprocessed_data_file_path)
    print("Read data for data analysis")
    '''
        Data Analysis Part
    '''
    analyse_data(data_for_analysis)
    print("Data Analysis done\n")

    '''
        KNN Part
    '''
    knn()
    print("KNN done\n")

    '''
        Linear Regression Part
    '''
    # # Degree 1 (Not useful to run as we are already running till degree 20 in second part so this is also covered)
    # linreg_data_path = "./data/external/linreg.csv"
    # linreg_data = np.genfromtxt(linreg_data_path, delimiter=',', skip_header=True)
    # linear_regression(linreg_data)

    # Degree > 1
    linreg_data_path = "./data/external/linreg.csv"
    linreg_data = np.genfromtxt(linreg_data_path, delimiter=',', skip_header=True)
    print("Read data for linear regression")
    linear_regression(linreg_data, max_k=20)
    print("Linear regression done\n")

    # L1 & L2 regularisation
    regularisation_data_path = "./data/external/regularisation.csv"
    regularisation_data = np.genfromtxt(regularisation_data_path, delimiter=',', skip_header=True)
    print("Read data for regularisation")
    '''
        For comparison we can do linear regression for the regularisation.csv data without regularisation
        For better visualisation of regularisation taking place, try to keep the training set size about 4-5%
    '''
    # =============================================================================
    # linear_regression(regularisation_data, 20)
    # =============================================================================

    linear_regression(regularisation_data, 20, "l1", 10)
    print("L1 regularisation done")
    linear_regression(regularisation_data, 20, "l2", 10)
    print("L2 regularisation done\n")