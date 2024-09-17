# pip3 install pandas pyarrow          # For reading the .feather file
import pandas as pd
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as hc
from sklearn.mixture import GaussianMixture
from wordcloud import WordCloud
from sklearn.metrics import silhouette_score
from matplotlib.animation import FuncAnimation
import imageio
from matplotlib.patches import Ellipse

# Colors for printing for better readability
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM
from models.pca.pca import PCA
from models.knn.knn import KNN

# Data params dictionaries contain all the parameters that are set manually through inspection
test_data_params = {
    "n_clusters":    3,         # For testing purposes
    "type":          "test data",
    "data_path":     "./data/external/temp_data_k_means.csv",
    "train_percent": 100,       # Unsupervised learning
    "test_percent":  0,
    "val_percent":   0,
    "shuffle":       False,
    "seed":          42,
    "max_k":         15,        # Decided upon the value of 15 after watching the graph till various points (max = 200) (no. of data points in data)
    "init_method":   "kmeans",  # Can be kmeans or kmeans++
    "k_kmeans1":     3,         # Based on visual inspection of the k vs MCSS plot deciding the number of clusters (elbow point)
    "k_gmm1":        3,         # Based on visual instpection AIC BIC plot
    "k2":            3,         # Based on visual inspection of the PCA plots deciding the number of clusters
    "op_dim":        2,         # Based on visual inspection of the scree plot deciding the number of optimal dimensions
    "k_kmeans3":     3,         # Based on visual inspection of the k vs MCSS plot on reduced dataset deciding the number of clusters (elbow point)
    "k_gmm3":        3,         # Based on visual instpection AIC BIC plot on reduced dataset
    "label_idx":     2,         # Column Index of the label in the data
    "k_kmeans":      3,         # Based on interpreting the clustered labels deciding the number of clusters
    "k_gmm":         3,         # Based on interpreting the clustered labels deciding the number of clusters
    "linkage":       "ward",    # Based on interpreting the dendrograms deciding the best linkage method
    "k_best1":       3,         # == k_kmeans
    "k_best2":       3,         # == k_gmm
}

original_data_params = {
    "n_clusters":    5, # For testing purposes
    "type":          "original assignment data",
    "data_path":     "./data/external/word-embeddings.feather",
    "train_percent": 100,       # Unsupervised learning
    "test_percent":  0,
    "val_percent":   0,
    "shuffle":       False,
    "seed":          42,
    "max_k":         15,        # Decided upon the value of 15 after watching the graph till various points (max = 200) (no. of data points in data)
    "init_method":   "kmeans++",  # Can be kmeans or kmeans++
    "k_kmeans1":     5,         # Based on visual inspection of the k vs MCSS plot deciding the number of clusters (elbow point)
    "k_gmm1":        1,         # Based on visual inspection of AIC BIC plot
    "k2":            3,         # Based on visual inspection of the PCA plots deciding the number of clusters
    "op_dim":        4,         # Based on visual inspection of the scree plot deciding the number of optimal dimensions
    "k_kmeans3":     6,         # Based on visual inspection of the k vs MCSS plot on reduced dataset deciding the number of clusters (elbow point)
    "k_gmm3":        4,         # Based on visual inspection of the AIC BIC plot on reduced dataset deciding the number of clusters
    "label_idx":     0,         # Column Index of the label in the data
    "k_kmeans":      3,         # Based on interpreting the clustered labels deciding the number of clusters
    "k_gmm":         4,         # Based on interpreting the clustered labels deciding the number of clusters
    "linkage":       "ward",    # Based on interpreting the dendrograms deciding the best linkage method
    "k_best1":       4,         # == k_kmeans
    "k_best2":       4,         # == k_gmm
}

spotify_data_params = {
    "pca_n_components" : 9, # Decided after examining the scree plot
    "file_path" : './data/external/spotify.csv' ,
    "preprocessed_file_path" : './data/interim/2/preprocessed_spotify',
    "best_k_spotify" : 50,
    "best_distance_metric_spotify" : "manhattan",
}

# Plots the word cloud for the given clusters
def plot_word_cloud(clusters, params, k=None, n_rows=2, n_cols=2, save_as="word_cloud", width=800, height=400, background_color='white', max_font_size=100, min_font_size=10, colormap='viridis', max_words=1000, collocations=False):
    """
        Plots the word cloud for the given clusters

        Parameters
        ==========
            clusters (list): List of clusters where each cluster is a list of words
            params (dict): Dictionary containing the parameters
            k (int): Number of clusters
            n_rows (int): Number of rows in the plot
            n_cols (int): Number of columns in the plot
            save_as (str): Name to save the plot
            width (int): Width of the plot
            height (int): Height of the plot
            background_color (str): Background color of the plot
            max_font_size (int): Maximum font size in the plot
            min_font_size (int): Minimum font size in the plot
            colormap (str): Colormap for the plot
            max_words (int): Maximum number of words to display
            collocations (bool): Whether to consider collocations or not

        Returns
        =======
            None
    """
    wordcloud = WordCloud(
        width=width, height=height,
        background_color=background_color,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        colormap=colormap,
        max_words=max_words,
        collocations=collocations
    )

    if n_rows == 1 and n_cols == 1:
        wordcloud.generate(' '.join(clusters[0]))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Cluster 1')
        plt.savefig(f'./assignments/2/figures/{save_as}.png')
        plt.close()
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 7))

    for cluster_idx, cluster in enumerate(clusters):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")
        row_idx = cluster_idx // n_cols
        col_idx = cluster_idx % n_cols
        wordcloud.generate(' '.join(cluster))
        axes[row_idx, col_idx].imshow(wordcloud, interpolation='bilinear')
        axes[row_idx, col_idx].axis('off')
        axes[row_idx, col_idx].set_title(f'\n\nCluster {cluster_idx + 1}')

    plt.suptitle(f"Word Clouds for {k} = {params[k]} Clusters")
    plt.tight_layout()
    plt.savefig(f'./assignments/2/figures/{save_as}.png')
    plt.close()

# Reads the data from the given path and returns it as a numpy array after processing
def read_data(data_path):
    """
        Reads the data from the given path and returns it as a numpy array after processing,
        Assuming that .csv is the test file and .feather is the original data file. Custom
        function for the particular dataset at hand - cannot be used for other datasets, unless
        the dataset is in the same format

        Parameters
        ==========
            data_path (str): Path to the data file

        Returns
        =======
            original_data (numpy array): The original data read from the file
            processed_data (numpy array): Data after removing the label column
    """
    if data_path.endswith('.csv'):
        # First 2 columns are data ('x' & 'y') and the last column is the label for the test dataset
        df = pd.read_csv(data_path)
        original_data = df.to_numpy()
        processed_data = []
        # Removing the last column which is the label and keeping only 'x' & 'y' in processed data
        for row in original_data:
            processed_data.append(row[:-1:].tolist())
        processed_data = np.array(processed_data)
    elif data_path.endswith('.feather'):
        # First column is the word, second column is the embedding (as a numpy array) of 512 length for word-embeddings.feather
        df = pd.read_feather(data_path)
        original_data = df.to_numpy()
        processed_data = []
        # Removing the first column which is the word and keeping only the embedding in processed data
        for row in original_data:
            processed_data.append(row[1].tolist())
        processed_data = np.array(processed_data)
    return original_data, processed_data

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

# Runs a KMeans model with the given inputs and prints statistics
def run_k_means_model(k, train_data, seed=None, init_method="kmeans", print_op=True):
    """
        Runs a KMeans model with the given inputs and prints statistics

        Parameters
        ==========
            k (int): Number of clusters
            train_data[n_train, n_features] (numpy array): Data to fit the model
            seed (int): Seed for shuffling
            init_method (str): Method to initialize the centroids (kmeans or kmeans++)
            print_op (bool): Whether to print the statistics or not

        Returns
        =======
            k_means_model (KMeans): The trained KMeans model
    """
    k_means_model = KMeans(k, init_method=init_method)
    k_means_model.load_data(train_data)

    # Fitting the model and recording time taken to fit along with the number of epochs taken
    start_time = time.time()
    epochs_taken = k_means_model.fit(seed)
    end_time = time.time()
    if print_op:
        print(f"\t{GREEN}Time taken to fit: {round(end_time - start_time, 5)} s{RESET}")
        print(f"\t{GREEN}Epochs taken to fit: {epochs_taken}{RESET}")
        final_cost = round(k_means_model.getCost(), 5)
        print(f"\t{GREEN}Final Cost: {final_cost}{RESET}")
        print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")
    return k_means_model

# Runs KMeans model for different k values and plots the elbow plot
def draw_elbow_plot(train_data, save_as, max_k=15, init_method="kmeans", seed=None):
    """
        Draws the elbow plot for the given data

        Parameters
        ==========
            train_data[n_train, n_features] (numpy array): Data to fit the model
            save_as (str): Name to save the plot
            max_k (int): Maximum value of k to consider
            init_method (str): Method to initialize the centroids
            seed (int): Seed for shuffling
        
        Returns
        =======
            None
    """

    '''
        Source: https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/

        Not always the best method: The elbow method might not be suitable for all datasets, 
        especially for those with high dimensionality or clusters of irregular shapes.
        Hence if you see the graph it is very difficult to identify the elbow point.
    '''

    # For each value of k, run the KMeans model and store the final cost (WCSS)
    k_arr = range(1, max_k + 1)
    final_cost_arr = []
    for k in k_arr:
        print(f"{BLUE}k = {k}{RESET}")
        trained_k_means_model = run_k_means_model(k, train_data, seed=seed, init_method=init_method)
        final_cost_arr.append(round(trained_k_means_model.getCost(), 5))

    # Plotting k vs final cost
    plt.figure(figsize=(10, 6))
    plt.plot(k_arr, final_cost_arr, 'o-')
    plt.xlabel('k')
    plt.ylabel('Final WCSS Cost')
    plt.title('k vs Final WCSS Cost')
    plt.grid(True)
    plt.savefig(f'./assignments/2/figures/k_means/{save_as}.png')
    plt.close()
    print(f"{GREEN}{save_as} plot saved{RESET}\n")

# Runs a scikit learn GMM model with the given inputs and prints statistics
def run_gmm_model(k, train_data, seed=None, model_type="sklearn", print_op=True):
    """
        Runs a GMM model with the given inputs and prints statistics

        Parameters
        ==========
            k (int): Number of clusters
            train_data[n_train, n_features] (numpy array): Data to fit the model
            seed (int): Seed for shuffling
            model_type (str): Type of model to use (sklearn or custom)
            print_op (bool): Whether to print the statistics or not

        Returns
        =======
            gmm_model (GaussianMixture): The trained GMM model
    """
    if model_type == "sklearn":
        if seed is None:
            gmm_model = GaussianMixture(k)
        else:
            gmm_model = GaussianMixture(k, random_state=seed)
        start_time = time.time()
        gmm_model.fit(train_data)
        gmm_model.predict(train_data)
        end_time = time.time()
        if print_op:
            print(f"\t{GREEN}Time taken to fit: {round(end_time - start_time, 5)} s{RESET}")
            overall_log_likelihood = round(gmm_model.score(train_data) * train_data.shape[0], 5)
            print(f"\t{GREEN}Final Log Likelihood: {overall_log_likelihood}{RESET}")
            print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")
        return gmm_model
    else:
        gmm_model = GMM(k, seed)
        gmm_model.load_data(train_data)
        start_time = time.time()
        gmm_model.fit()
        end_time = time.time()
        if print_op:
            print(f"\t{GREEN}Time taken to fit: {round(end_time - start_time, 5)} s{RESET}")
            overall_log_likelihood = round(gmm_model.get_log_likelihood(), 5)
            print(f"\t{GREEN}Final Log Likelihood: {overall_log_likelihood}{RESET}")
            print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")
        return gmm_model

# Implements AIB and BIC graph for scikit-learn GMM
def gmm_aic_bic_method(max_k, train_data, save_as, seed=None, model_type="sklearn"):
    """
        Implements AIB and BIC graph for GMM

        Parameters
        ==========
            max_k (int): Maximum value of k to consider
            train_data[n_train, n_features] (numpy array): Data to fit the model
            save_as (str): Name to save the plot
            seed (int): Seed for shuffling
            model_type (str): Type of model to use (sklearn or custom)
        
        Returns
        =======
            None
    """

    # Running GMM model for different k values and storing AIC and BIC values
    k_arr = range(1, max_k + 1)
    aic_arr = []
    bic_arr = []
    for k in k_arr:
        print(f"{BLUE}k = {k}{RESET}") 
        trained_gmm_model = run_gmm_model(k, train_data, seed, model_type)
        if model_type == "sklearn":
            aic_arr.append(trained_gmm_model.aic(train_data))
            bic_arr.append(trained_gmm_model.bic(train_data))
        else:
            overall_log_likelihood = trained_gmm_model.get_log_likelihood()
            aic = (2 * k) - (2 * overall_log_likelihood)
            bic = (k * np.log(train_data.shape[0])) - (2 * overall_log_likelihood)
            aic_arr.append(aic)
            bic_arr.append(bic)

    # Plotting k vs aic and bic
    plt.figure(figsize=(10, 6))
    plt.plot(k_arr, aic_arr, 'o-', label='AIC', color='red')
    plt.plot(k_arr, bic_arr, 'o-', label='BIC', color='blue')
    plt.xlabel('k')
    plt.grid(True)
    plt.ylabel('AIC/BIC')
    plt.legend()
    plt.title('k vs AIC/BIC')
    plt.savefig(f'./assignments/2/figures/gmm/{save_as}.png')
    plt.close()
    print(f"{GREEN}{save_as} plot saved{RESET}\n")

# Implements the PCA algorithm
def run_pca_model(n_components, data):
    """
        Implements the PCA algorithm and plots the transformed data for 2 and 3 components

        Parameters
        ==========
            n_components (int): Number of components to reduce to
            data[n_points, n_features] (numpy array): Data to fit the model

        Returns
        =======
            transformed_data[n_points, n_components] (numpy array): Transformed data
    """
    # Running PCA model
    pca_model = PCA(n_components)
    pca_model.load_data(data)
    pca_model.fit()
    # Transforming the data
    transformed_data = pca_model.transform()

    # Checking if PCA transformation was successful
    if pca_model.checkPCA(transformed_data):
        print(f"{GREEN}PCA transformation for {n_components} components successful{RESET}")
    else:
        print(f"{RED}PCA transformation for {n_components} unsuccessful{RESET}")
        raise(ValueError)
    
    # Plotting the transformed data
    if n_components == 2:
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Principal Component Analysis')
        plt.savefig('./assignments/2/figures/pca/pca_2.png')
        plt.close()
        print(f"{GREEN}PCA plot saved{RESET}\n")

    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title('Principal Component Analysis')
        '''
            The following code for creating gif animation is given by ChatGPT
            Prompt: Given a 3D plot in matplotlib generate a gif of the plot rotating it
        '''
        # ==============================================================================
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return ax,

        frames = np.arange(0, 360, 2)
        ani = FuncAnimation(fig, update, frames=frames, interval=50)
        ani.save('./assignments/2/figures/pca/pca_3.gif', writer='pillow')
        # ==============================================================================
        plt.close()
        print(f"{GREEN}PCA plot saved{RESET}\n")
    return transformed_data

# Plots scree plot and cumulative scree plot for deciding optimal number of dimensions for PCA
def draw_scree_plot(train_data, save_as):
    """
        Plots scree plot and cumulative scree plot for deciding optimal number of dimensions for PCA

        Parameters
        ==========
            train_data[n_train, n_features] (numpy array): Data to fit the model
            save_as (str): Name to save the plot

        Returns
        =======
            None
    """
    # Running PCA model
    pca_model = PCA()
    pca_model.load_data(train_data) 
    eigenvalues, eigenvectors = pca_model.get_eigenvalues_eigenvectors() # Gets all the eigenvalues and eigenvectors
    max_val = 21                                           # Max number of principal components to consider for plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x_axis = range(1, len(eigenvalues) + 1)
    # Plotting the scree plot
    axes[0].plot(x_axis[:max_val], eigenvalues[:max_val], 'o-', color='red')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Scree Plot')

    # Plotting cumulative eigenvalue plot
    total_sum = sum(eigenvalues)
    x_axis = range(1, len(eigenvalues) + 1)
    y_axis = [total_sum]
    for eigenvalue in eigenvalues[:-1:]:
        y_axis.append(total_sum - eigenvalue)
        total_sum -= eigenvalue
    axes[1].plot(x_axis[:max_val], y_axis[:max_val], 'o-', color='blue')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Eigenvalue')
    axes[1].set_title('Cumulative Scree Plot')
    plt.savefig(f'./assignments/2/figures/{save_as}')
    plt.close()
    print(f"{GREEN}Scree plot saved{RESET}\n")

# Given the cluster assignment, original data and the label index (index of column which represents label), returns the clusters of labels
def get_cluster_values(k, cluster_assignment, original_data, label_idx):
    """
        Given the cluster assignment, original data and the label index (index of column which represents label),
        returns the clusters corresponding to each label

        Parameters
        ==========
            k (int): Number of clusters
            cluster_assignment[n_points] (list): Cluster assignment for each data point
            original_data[n_points, n_features] (numpy array): Original data
            label_idx (int): Index of column which represents label
        
        Returns
        =======
            clusters (list): List of clusters where each cluster is a list of labels
    """
    # Normalizing the cluster numbers to start from 0 since the numbers returned by hierarchical clustering start from 1
    min_val = min(cluster_assignment)
    cluster_assignment = [val - min_val for val in cluster_assignment]

    clusters = [list() for _ in range(k)]
    for data_point_idx in range(len(cluster_assignment)):
        # For each data point determining the cluster number and the label
        cluster_no = cluster_assignment[data_point_idx]
        cluster_label = original_data[data_point_idx][label_idx]
        # If it is not a string, convert it to a string (in case of test data the label is a number)
        if type(cluster_label) != str:
            cluster_label = str(cluster_label)
        clusters[cluster_no].append(cluster_label)
    return clusters

# Plots the dendrogram for hierarchical clustering
def plot_dendrogram(type, original_data, preprocessed_data, method, metric, save_as):
    """
        Plots the dendrogram for hierarchical clustering

        Parameters
        ==========
            type (str): Type of data (original assignment data or test data)
            original_data[n_points, n_features] (numpy array): Original data
            preprocessed_data[n_points, n_features] (numpy array): Preprocessed data
            method (str): Method to use for hierarchical clustering
            metric (str): Metric to use for hierarchical clustering
            save_as (str): Name to save the plot
        
        Returns
        =======
            None
    """
    Z = hc.linkage(preprocessed_data, method=method, metric=metric)
    if type == "original assignment data":
        custom_labels = original_data[:, 0].tolist()
    elif type == "test data":
        custom_labels = original_data[:, 2].tolist()
    fig = plt.figure(figsize=(25, 10))
    dn = hc.dendrogram(Z, labels=custom_labels)
    plt.savefig(f'./assignments/2/figures/hierarchical_clustering/{save_as}.png')
    plt.close()
    print(f"{GREEN}{save_as} plot saved{RESET}\n")

# For preprocessing the spotify dataset
def preprocess_spotify_dataset(data, preprocessed_data_file_path):
    """
        Preprocesses the spotify dataset by performing the following steps:
            1. Renaming the first unnamed column as index
            2. Remove exact duplicates - if track id is same, the song is exactly the same
            3. Remove duplicate songs in multiple albums
            4. Z-score normalization
            5. Removing unnecessary columns and non-numeric columns
            6. Storing the preprocessed data for future use
        
        Parameters
        ==========
            data (pandas DataFrame): The original data
            preprocessed_data_file_path (str): Path to store the preprocessed data
        
        Returns
        =======
            data_with_track_genre (pandas DataFrame): Preprocessed data with track genre
            data_without_track_genre (pandas DataFrame): Preprocessed data without track genre
    """
    # Renaming the first unnamed column as index
    data = data.rename(columns={'Unnamed: 0': 'index'})

    # Remove exact duplicates - if track id is same, the song is exactly the same
    data = data.drop_duplicates("track_id")

    # Remove duplicate songs in multiple albums
    data = data.drop_duplicates("track_name")

    # Z-score normalization
    cols_to_normalise = ["popularity", "duration_ms", "danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    for col in cols_to_normalise:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std

    # Removing unnecessary columns and non-numeric columns
    data2 = data.copy()
    unnecessary_cols = ["index", "track_id", "album_name", "artists", "track_name", "explicit"]
    data_with_track_genre = data2.drop(columns=unnecessary_cols)

    unnecessary_cols = ["index", "track_id", "album_name", "artists", "track_name", "explicit", "track_genre"]
    data_without_track_genre = data.drop(columns=unnecessary_cols)

    # Storing the preprocessed data for future use
    data_with_track_genre.to_csv(preprocessed_data_file_path+"_with_track_genre.csv", index=False)
    data_without_track_genre.to_csv(preprocessed_data_file_path+"_without_track_genre.csv", index=False)

    return data_with_track_genre, data_without_track_genre

def spotify_dataset():
    """
        Implements the Spotify dataset analysis by performing the following steps:
            1. Preprocess the data
            2. Draw the scree plot
            3. Reduce the dataset to optimal number of dimensions
            4. Run KNN model on the reduced dataset
            5. Run KNN model on the complete dataset
        
        Parameters
        ==========
            None
        
        Returns
        =======
            None
    """
    try:
        # Checking if preprocessed data is already present
        spotify_data_with_track_genre = pd.read_csv(spotify_data_params["preprocessed_file_path"] + "_with_track_genre.csv", quotechar='"')
        spotify_data_without_track_genre = pd.read_csv(spotify_data_params["preprocessed_file_path"] + "_without_track_genre.csv", quotechar='"')
    except FileNotFoundError:
        spotify_data = pd.read_csv(spotify_data_params["file_path"], quotechar='"')
        spotify_data_with_track_genre, spotify_data_without_track_genre = preprocess_spotify_dataset(spotify_data, spotify_data_params["preprocessed_file_path"])
    
    header_with_track_genre = spotify_data_with_track_genre.columns.tolist()
    header_without_track_genre = spotify_data_without_track_genre.columns.tolist()

    spotify_data_with_track_genre = spotify_data_with_track_genre.to_numpy()
    spotify_data_without_track_genre = spotify_data_without_track_genre.to_numpy()
    draw_scree_plot(spotify_data_without_track_genre, save_as="spotify_dataset/scree_plot_9_1.png")

    # Dimentionality reduction based on the optimal number of dimensions determined above
    print(f"{MAGENTA}spotify_n_components = {spotify_data_params['pca_n_components']}{RESET}\n") # On interpreting the scree plot for the spotify dataset we identify the optimal number of dimensions
    reduced_dataset_header_without_track_genre = [f"component_{i + 1}" for i in range(spotify_data_params["pca_n_components"])]
    reduced_dataset_header_with_track_genre = reduced_dataset_header_without_track_genre.copy()
    reduced_dataset_header_with_track_genre.append(header_with_track_genre[-1])

    pca_model_spotify = PCA(spotify_data_params["pca_n_components"])
    pca_model_spotify.load_data(spotify_data_without_track_genre)
    pca_model_spotify.fit()
    reduced_dataset_spotify_without_track_genre = pca_model_spotify.transform()
    print(f"{GREEN}Dataset reduced to {spotify_data_params['pca_n_components']} dimensions{RESET}")

    # Associating reduced dataset with corresponding labels
    reduced_dataset_spotify_with_track_genre = np.hstack((reduced_dataset_spotify_without_track_genre, spotify_data_with_track_genre[:, -1:]))

    # Running KNN model on the reduced dataset
    train_data, test_data, val_data = split_data(reduced_dataset_spotify_with_track_genre, 80, 10, shuffle=True)
    knn_model = KNN(spotify_data_params["best_k_spotify"], spotify_data_params["best_distance_metric_spotify"])
    knn_model.load_train_test_val_data(reduced_dataset_header_with_track_genre, train_data, test_data, val_data)
    knn_model.set_predict_var("track_genre")
    knn_model.use_for_prediction(reduced_dataset_header_without_track_genre)
    start_time = time.time()
    knn_model.fit()
    knn_model.predict("validation")
    metrics = knn_model.get_metrics()
    end_time = time.time()
    time_diff = end_time - start_time

    # Printing metrics
    print(f"""
    {BLUE}k = {spotify_data_params["best_k_spotify"]}, Distance Metric = {spotify_data_params["best_distance_metric_spotify"]}
    Validation Metrics for Reduced Spotify Dataset{RESET}\n
                {GREEN}Accuracy:        {round(metrics['accuracy'] * 100, 3)}%\n
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
    ---------------------------------------------------------------------{RESET}\n""")
    reduced_dataset_time = round(time_diff, 4)

    # Running KNN model on the complete dataset
    train_data, test_data, val_data = split_data(spotify_data_with_track_genre, 80, 10, shuffle=True)
    knn_model = KNN(spotify_data_params["best_k_spotify"], spotify_data_params["best_distance_metric_spotify"])
    knn_model.load_train_test_val_data(header_with_track_genre, train_data, test_data, val_data)
    knn_model.set_predict_var("track_genre")
    knn_model.use_for_prediction(header_without_track_genre)
    start_time = time.time()
    knn_model.fit()
    knn_model.predict("validation")
    metrics = knn_model.get_metrics()
    end_time = time.time()
    time_diff = end_time - start_time

    # Printing metrics
    print(f"""
    {BLUE}k = {spotify_data_params["best_k_spotify"]}, Distance Metric = {spotify_data_params["best_distance_metric_spotify"]}
    Validation Metrics for Complete Spotify Dataset{RESET}\n
                {GREEN}Accuracy:        {round(metrics['accuracy'] * 100, 3)}%\n
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
    ---------------------------------------------------------------------{RESET}\n""")
    complete_dataset_time = round(time_diff, 4)

    # Plotting the inference time comparison
    x_axis = ["Reduced Dataset", "Complete Dataset"]
    y_axis = [reduced_dataset_time, complete_dataset_time]
    plt.figure(figsize=(10, 6))
    plt.bar(x_axis, y_axis)
    plt.xlabel('Dataset')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time Comparison')
    plt.savefig('./assignments/2/figures/spotify_dataset/inference_time_comparison.png')
    plt.close()
    print(f"{GREEN}Inference Time Comparison plot saved{RESET}\n")

def visualise_hierarchical_clusters(train_data, clusters, save_as=None):
    pca_model = PCA(2)
    pca_model.load_data(train_data)
    pca_model.fit()
    transformed_data = pca_model.transform()
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Hierarchical Clustering')
    if save_as is not None:
        plt.savefig(f'./assignments/2/figures/hierarchical_clustering/{save_as}_pca_2.png')
    else:
        plt.show()
    plt.close()

    print(f"{GREEN}Hierarchical Clustering plot PCA 2 saved{RESET}\n")

    pca_model = PCA(3)
    pca_model.load_data(train_data)
    pca_model.fit()
    transformed_data = pca_model.transform()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=clusters, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('Hierarchical Clustering')
    if save_as is not None:
        '''
            The following code for creating gif animation is given by ChatGPT
            Prompt: Given a 3D plot in matplotlib generate a gif of the plot rotating it
        '''
        # ==============================================================================
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return ax,

        frames = np.arange(0, 360, 2)
        ani = FuncAnimation(fig, update, frames=frames, interval=50)
        ani.save(f'./assignments/2/figures/hierarchical_clustering/{save_as}_pca_3.gif', writer='pillow')
        # ==============================================================================
    else:
        plt.show()
    plt.close()

    print(f"{GREEN}Hierarchical Clustering plot PCA 3 saved{RESET}\n")

def hierarchical_clustering(params, original_data, train_data):
    """
        Implements the Hierarchical Clustering algorithm by performing the following steps:
            1. Plot the dendrogram for different linkage methods and distance metrics
            2. Infer the best linkage method
            3. Cluster the data for the best linkage method and distance metric
            4. Plot the word cloud for the clusters
        
        Parameters
        ==========
            params (dict): Dictionary containing the parameters
            original_data[n_points, n_features] (numpy array): Original data
            train_data[n_train, n_features] (numpy array): Data for training
        
        Returns
        =======
            None
    """
    distance_metrics=['euclidean', 'cityblock', 'cosine'] # cityblock is same as manhattan distance metric
    methods=['single', 'complete', 'average', 'ward', 'centroid', 'median']

    for method in methods:
        if method == "ward" or method == "centroid" or method == "median": # Ward, centroid and median method works only with euclidean distance metric
            plot_dendrogram(params["type"], original_data, train_data, method=method, metric='euclidean', save_as=f'dendrogram_{method}_euclidean')
            continue
        for metric in distance_metrics:
            plot_dendrogram(params["type"], original_data, train_data, method=method, metric=metric, save_as=f'dendrogram_{method}_{metric}')

    # On interpreting the dendrograms we identify the best linkage method
    print(f"{MAGENTA}Inferred best Linkage method = {params["linkage"]}{RESET}\n")

    Z = hc.linkage(train_data, method=params["linkage"], metric='euclidean')
    # For kbest1 (K-Means)
    clusters_kbest1 = hc.fcluster(Z, t=params["k_best1"], criterion='maxclust')
    visualise_hierarchical_clusters(train_data, clusters_kbest1, save_as="k_best1")

    # For kbest2 (GMM)
    clusters_kbest2 = hc.fcluster(Z, t=params["k_best2"], criterion='maxclust')
    visualise_hierarchical_clusters(train_data, clusters_kbest2, save_as="k_best2")

    clustered_values_kbest1 = get_cluster_values(params["k_best1"], clusters_kbest1, original_data, params["label_idx"])
    clustered_values_kbest2 = get_cluster_values(params["k_best2"], clusters_kbest2, original_data, params["label_idx"])
    
    print(f"{BLUE}Clustered labels for k_best1 = {params["k_best1"]}{RESET}")
    for cluster_idx, cluster in enumerate(clustered_values_kbest1):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")

    plot_word_cloud(clustered_values_kbest1, params, "k_best1", save_as="hierarchical_clustering/word_cloud_k_best1")
    print(f"{GREEN}Word Cloud for k_best1 saved{RESET}\n")

    print(f"{BLUE}Clustered labels for k_best2 = {params["k_best2"]}{RESET}")
    for cluster_idx, cluster in enumerate(clustered_values_kbest2):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")

    plot_word_cloud(clustered_values_kbest2, params, "k_best2", save_as="hierarchical_clustering/word_cloud_k_best2", colormap="winter")
    print(f"{GREEN}Word Cloud for k_best2 saved{RESET}\n")

def draw_individual_component_labelled_scatter_plot(original_data, transformed_data, save_as):
    """
        Draws the loading plot for PCA

        Parameters
        ==========
            original_data[n_points, n_features] (numpy array): Original data
            transformed_data[n_points, n_components] (numpy array): Transformed data
        
        Returns
        =======
            None
    """
    fig, axes = plt.subplots(transformed_data.shape[1], 1, figsize=(15, 8))

    for component_idx in range(transformed_data.shape[1]):
        axes[component_idx].scatter(transformed_data[:, component_idx], np.zeros_like(transformed_data[:, component_idx]), color='blue', s=2)
        for data_point_idx, data_point in enumerate(transformed_data):
            label = original_data[data_point_idx][0]
            axes[component_idx].text(data_point[component_idx], 0, label, rotation=90, fontsize=6)
        axes[component_idx].set_xlabel(f'Principal Component {component_idx + 1}')
    plt.savefig(f'./assignments/2/figures/pca/{save_as}.png')
    plt.close()
    print(f"{GREEN}{save_as} plot saved{RESET}\n")

def plot_k_means_clusters(transformed_data, k, save_as=None):
    """
        Plots the KMeans Clustering (Image for 2D and GIF for 3D)

        Parameters
        ==========
            transformed_data[n_points, n_components] (numpy array): Transformed data
            k (int): Number of clusters
            save_as (str): Name to save the plot
        
        Returns
        =======
            None
    """
    k_means_model = KMeans(k, init_method="kmeans++")
    k_means_model.load_data(transformed_data)
    k_means_model.fit()
    k_means_model.predict(transformed_data)
    if transformed_data.shape[1] == 2:
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=k_means_model.cluster_labels)
        plt.scatter(k_means_model.cluster_centers[:, 0], k_means_model.cluster_centers[:, 1], c='red', s=100, marker='x')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('KMeans Clustering')
        if save_as is not None:
            plt.savefig(f'./assignments/2/figures/k_means/{save_as}.png')
        else:
            plt.show()
    elif transformed_data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=k_means_model.cluster_labels)
        ax.scatter(k_means_model.cluster_centers[:, 0], k_means_model.cluster_centers[:, 1], k_means_model.cluster_centers[:, 2], c='red', s=100, marker='x')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title('KMeans Clustering')
        if save_as is not None:
            '''
                The following code for creating gif animation is given by ChatGPT
                Prompt: Given a 3D plot in matplotlib generate a gif of the plot rotating it
            '''
            # ==============================================================================
            def update(frame):
                ax.view_init(elev=30, azim=frame)
                return ax,

            frames = np.arange(0, 360, 2)
            ani = FuncAnimation(fig, update, frames=frames, interval=50)
            ani.save(f'./assignments/2/figures/k_means/{save_as}.gif', writer='pillow')
            # ==============================================================================
        else:
            plt.show()
    
    plt.close()
    print(f"{GREEN}KMeans Clustering plot for k = {k} saved{RESET}\n")

# Runs PCA to reduce the dimensions to 2 and 3 for visual analysis
def pca(original_data, train_data, save=False):
    """
        Runs PCA to reduce the dimensions to 2 and 3 for visual analysis and determining the value of k2

        Parameters
        ==========
            train_data[n_train, n_features] (numpy array): Data to fit the model

        Returns
        =======
            None
    """
    n_components = 2
    transformed_data = run_pca_model(n_components, train_data)
    draw_individual_component_labelled_scatter_plot(original_data, transformed_data, save_as="component_wise_labelled_scatter_plot_2")
    if save == True:
        plot_k_means_clusters(transformed_data, 3, save_as="visualising_clusters_pca_2_k_3")
        plot_k_means_clusters(transformed_data, 4, save_as="visualising_clusters_pca_2_k_4")
    else:
        plot_k_means_clusters(transformed_data, 3)
        plot_k_means_clusters(transformed_data, 4)

    n_components = 3
    transformed_data = run_pca_model(n_components, train_data)
    draw_individual_component_labelled_scatter_plot(original_data, transformed_data, save_as="component_wise_labelled_scatter_plot_3")
    if save == True:
        plot_k_means_clusters(transformed_data, 3, save_as="visualising_clusters_pca_3_k_3")
        plot_k_means_clusters(transformed_data, 4, save_as="visualising_clusters_pca_3_k_4")
    else:
        plot_k_means_clusters(transformed_data, 3)
        plot_k_means_clusters(transformed_data, 4)

def k_means(params, train_data):
    """
        Implements the KMeans algorithm, draws elbow plot and decides value for k_kmeans1

        Parameters
        ==========
            params (dict): Dictionary containing all the parameters
            train_data[n_train, n_features] (numpy array): Data to fit the model

        Returns
        =======
            None
    """
    # Drawing the elbow plot
    draw_elbow_plot(train_data, save_as="elbow_plot_full_dataset", max_k=params["max_k"], init_method=params["init_method"], seed=params["seed"])

    # On interpreting the elbow plot we identify the value of k_kmeans1
    print(f"{MAGENTA}Inferred value of k_kmeans1 = {params["k_kmeans1"]}{RESET}\n")

    # Running the KMeans model with the value of k_kmeans1
    print(f"{BLUE}KMeans with k = {params["k_kmeans1"]}{RESET}\n")
    run_k_means_model(params["k_kmeans1"], train_data, seed=params["seed"], init_method=params["init_method"])

def k_means_cluster_analysis(params, train_data, original_data):
    """
        Implements the KMeans Cluster Analysis by performing the following steps:
            1. Cluster Analysis for different values of k
            2. Plotting inertial scores and silhouette scores

        Parameters
        ==========
            params (dict): Dictionary containing all the parameters
            train_data[n_train, n_features] (numpy array): Data to fit the model
            original_data[n_points, n_features] (numpy array): Original data
        
        Returns
        =======
            None
    """
    inertial_scores = list()
    silhouette_scores = list()
    
    trained_model_k_kmeans1 = run_k_means_model(params["k_kmeans1"], train_data, print_op=False, seed=params["seed"], init_method=params["init_method"])

    inertial_scores.append(trained_model_k_kmeans1.getCost())
    silhouette_scores.append(silhouette_score(train_data, trained_model_k_kmeans1.cluster_labels))
    
    clustered_values_k_kmeans1 = get_cluster_values(params["k_kmeans1"], trained_model_k_kmeans1.cluster_labels, original_data, params["label_idx"])
    print(f"{BLUE}Clustered labels for k_kmeans1 = {params["k_kmeans1"]}{RESET}")
    for cluster_idx, cluster in enumerate(clustered_values_k_kmeans1):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")

    plot_word_cloud(clustered_values_k_kmeans1, params, "k_kmeans1", n_rows=3, n_cols=2, save_as="cluster_analysis/word_cloud_k_kmeans1", colormap="plasma")

    print("------------------------------------------------------------------------------------------------------")

    trained_model_k2 = run_k_means_model(params["k2"], train_data, print_op=False, seed=params["seed"], init_method=params["init_method"])

    inertial_scores.append(trained_model_k2.getCost())
    silhouette_scores.append(silhouette_score(train_data, trained_model_k2.cluster_labels))

    clustered_values_k2 = get_cluster_values(params["k2"], trained_model_k2.cluster_labels, original_data, params["label_idx"])
    print(f"{BLUE}Clustered labels for k2 = {params["k2"]}{RESET}")
    for cluster_idx, cluster in enumerate(clustered_values_k2):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")

    plot_word_cloud(clustered_values_k2, params, "k2", save_as="cluster_analysis/word_cloud_k2_kmeans", colormap="winter")

    print("------------------------------------------------------------------------------------------------------")

    trained_model_k_kmeans3 = run_k_means_model(params["k_kmeans3"], train_data, print_op=False, seed=params["seed"], init_method=params["init_method"])

    inertial_scores.append(trained_model_k_kmeans3.getCost())
    silhouette_scores.append(silhouette_score(train_data, trained_model_k_kmeans3.cluster_labels))

    clustered_values_k_kmeans3 = get_cluster_values(params["k_kmeans3"], trained_model_k_kmeans3.cluster_labels, original_data, params["label_idx"])
    print(f"{BLUE}Clustered labels for k_kmeans3 = {params["k_kmeans3"]}{RESET}")
    for cluster_idx, cluster in enumerate(clustered_values_k_kmeans3):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")

    plot_word_cloud(clustered_values_k_kmeans3, params, "k_kmeans3", n_rows=2, n_cols=3, save_as="cluster_analysis/word_cloud_k_kmeans3")

    print("------------------------------------------------------------------------------------------------------")

    # Plotting inertial scores and silhouette scores
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    x_axis = np.array([params["k_kmeans1"], params["k2"], params["k_kmeans3"]])
    idxs = np.argsort(x_axis)
    x_axis = x_axis[idxs]
    y_axis1 = np.array(inertial_scores)[idxs]
    y_axis2 = np.array(silhouette_scores)[idxs]

    axes[0].plot(x_axis, y_axis1, 'o-', label='Inertial Scores', color='red')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia Scores')
    axes[0].set_title('k vs Inertia Scores')
    axes[0].grid(True)
    axes[1].plot(x_axis, y_axis2, 'o-', label='Silhouette Scores', color='blue')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette Scores')
    axes[1].set_title('k vs Silhouette Scores')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('./assignments/2/figures/cluster_analysis/k_vs_scores_kmeans.png')
    plt.close()
    print(f"{GREEN}k vs Scores plot saved{RESET}\n")

def gmm_cluster_analysis(params, train_data, original_data, model_type="sklearn"):
    """
        Implements the GMM Cluster Analysis by performing the following steps:
            1. Cluster Analysis for different values of k
            2. Plotting word clouds for the clusters

        Parameters
        ==========
            params (dict): Dictionary containing all the parameters
            train_data[n_train, n_features] (numpy array): Data to fit the model
            original_data[n_points, n_features] (numpy array): Original data
            model_type (str): Type of model to use (my or sklearn)

        Returns
        =======
            None
    """
    inertial_scores = list()
    silhouette_scores = list()

    def compute_inertia(X, labels, means):
        inertia = 0
        for i in range(len(means)):
            cluster_points = X[labels == i]
            inertia += np.sum((cluster_points - means[i]) ** 2)
        return inertia

    trained_model_k_gmm1 = run_gmm_model(params["k_gmm1"], train_data, seed=params["seed"], model_type=model_type, print_op=False)
    
    
    if model_type == "sklearn":
        inertial_scores.append(compute_inertia(train_data, trained_model_k_gmm1.predict(train_data), trained_model_k_gmm1.means_))
        if params["k_gmm1"] > 1:
            silhouette_scores.append(silhouette_score(train_data, trained_model_k_gmm1.predict(train_data)))
        else:
            silhouette_scores.append(0)
    predicted_clusters = [int(np.argmax(cluster)) for cluster in trained_model_k_gmm1.predict_proba(train_data)]

    clustered_values_k_gmm1 = get_cluster_values(params["k_gmm1"], predicted_clusters, original_data, params["label_idx"])
    print(f"{BLUE}Clustered labels for k_gmm1 = {params["k_gmm1"]}{RESET}")
    print(f"\n{GREEN}Cluster 1{RESET}:")
    print(f"{clustered_values_k_gmm1[0]}\n")

    plot_word_cloud(clustered_values_k_gmm1, params, "k_gmm1", width=1000, height=500, n_rows=1, n_cols=1, save_as="cluster_analysis/word_cloud_k_gmm1", colormap="plasma")

    print("------------------------------------------------------------------------------------------------------")

    trained_model_k2 = run_gmm_model(params["k2"], train_data, seed=params["seed"], model_type=model_type, print_op=False)

    if model_type == "sklearn":
        inertial_scores.append(compute_inertia(train_data, trained_model_k2.predict(train_data), trained_model_k2.means_))
        silhouette_scores.append(silhouette_score(train_data, trained_model_k2.predict(train_data)))

    predicted_clusters = [int(np.argmax(cluster)) for cluster in trained_model_k2.predict_proba(train_data)]
    clustered_values_k2 = get_cluster_values(params["k2"], predicted_clusters, original_data, params["label_idx"])
    print(f"{BLUE}Clustered labels for k2 = {params["k2"]}{RESET}")
    for cluster_idx, cluster in enumerate(clustered_values_k2):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")
    plot_word_cloud(clustered_values_k2, params, "k2", save_as="cluster_analysis/word_cloud_k2_gmm", colormap="winter")

    print("------------------------------------------------------------------------------------------------------")

    trained_model_k_gmm3 = run_gmm_model(params["k_gmm3"], train_data, seed=params["seed"], model_type=model_type, print_op=False)

    if model_type == "sklearn":
        inertial_scores.append(compute_inertia(train_data, trained_model_k_gmm3.predict(train_data), trained_model_k_gmm3.means_))
        silhouette_scores.append(silhouette_score(train_data, trained_model_k_gmm3.predict(train_data)))

    predicted_clusters = [int(np.argmax(cluster)) for cluster in trained_model_k_gmm3.predict_proba(train_data)]
    clustered_values_k_gmm3 = get_cluster_values(params["k_gmm3"], predicted_clusters, original_data, params["label_idx"])
    print(f"{BLUE}Clustered labels for k_gmm3 = {params["k_gmm3"]}{RESET}")
    for cluster_idx, cluster in enumerate(clustered_values_k_gmm3):
        print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
        print(f"{cluster}\n")
    
    plot_word_cloud(clustered_values_k_gmm3, params, "k_gmm3", save_as="cluster_analysis/word_cloud_k_gmm3", colormap="viridis")

    print("------------------------------------------------------------------------------------------------------")

    # On interpreting the clusters we identify the value of k_gmm
    print(f"{MAGENTA}k_gmm = {params["k_gmm"]}{RESET}\n")

    # Plotting inertial scores and silhouette scores
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    x_axis = np.array([params["k_gmm1"], params["k2"], params["k_gmm3"]])
    idxs = np.argsort(x_axis)
    x_axis = x_axis[idxs]
    y_axis1 = np.array(inertial_scores)[idxs]
    y_axis2 = np.array(silhouette_scores)[idxs]

    axes[0].plot(x_axis, y_axis1, 'o-', label='Inertial Scores', color='red')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia Scores')
    axes[0].set_title('k vs Inertia Scores')
    axes[0].grid(True)
    axes[1].plot(x_axis, y_axis2, 'o-', label='Silhouette Scores', color='blue')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette Scores')
    axes[1].set_title('k vs Silhouette Scores')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('./assignments/2/figures/cluster_analysis/k_vs_scores_gmm.png')
    plt.close()
    print(f"{GREEN}k vs Scores plot saved{RESET}\n")

def plot_gmm_clusters(transformed_data, k, save_as):
    '''
        The following code for plotting and visualising GMM clusters is given by ChatGPT
        Prompt: Given a 2D plot of data points, plot the GMM clusters
    '''
    # ==============================================================================
    # Plot the GMM ellipses
    def plot_gmm_ellipses(gmm, ax):
        for i in range(gmm.n_components):
            # Get the mean and covariance matrix of the ith Gaussian component
            mean = gmm.means_[i]
            cov = gmm.covariances_[i]

            # Calculate the eigenvalues and eigenvectors
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Adjust for ellipse width and height
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0]) * 180.0 / np.pi

            # Create and add the ellipse to the plot
            ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color='r', alpha=0.3)
            ax.add_patch(ell)
    
    gmm = GaussianMixture(n_components=k, random_state=0)
    gmm.fit(transformed_data)

    # Predict the cluster labels
    labels = gmm.predict(transformed_data)

    # Plotting
    plt.figure(figsize=(10, 8))

    # Scatter plot of data points with cluster labels
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', marker='o', s=50, edgecolor='k')
    plt.title('GMM Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plot_gmm_ellipses(gmm, plt.gca())

    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.savefig(f'./assignments/2/figures/gmm/{save_as}.png')
    plt.close()
    # ==============================================================================

def compare_kmeans_gmm(params, train_data, original_data):
    pca_model = PCA(2)
    pca_model.load_data(train_data)
    pca_model.fit()
    transformed_data = pca_model.transform()
    plot_k_means_clusters(transformed_data, params["k_kmeans"], save_as="../cluster_analysis/kmeans_k_kmeans")
    plot_gmm_clusters(transformed_data, params["k_gmm"], save_as="../cluster_analysis/gmm_k_gmm")
    print(f"{GREEN}Comparison plots between K-Means and GMM plotted{RESET}")

if __name__ == "__main__":
    params = original_data_params

    # Original Data contains label column as well and Preprocessed Data does not contain the label column
    original_data, preprocessed_data = read_data(params["data_path"])
    # Data Splitting
    train_data, test_data, val_data = split_data(preprocessed_data, params["train_percent"], params["test_percent"], shuffle=params["shuffle"])

    # =================================================================
    #                           3.1 & 3.2   
    # =================================================================
    k_means(params, train_data)

    # =================================================================
    #                           4.1 & 4.2   
    # =================================================================
    try:
        print("Running my GMM class for arbitrary value of k")
        raise(ValueError)
        gmm_model = GMM(10)
        gmm_model.load_data(train_data)
        gmm_model.fit()
        overall_log_likelihood = gmm_model.get_log_likelihood()

        print(f"{GREEN}My GMM Class worked{RESET}")
        print("Plotting AIC BIC graph using my GMM Class\n")

        gmm_aic_bic_method(params["max_k"], train_data, save_as="aic_bic_plot_my_class", seed=params["seed"], model_type="my")

        print(f"{MAGENTA}Inferred value of k_gmm1 = {params["k_gmm1"]}{RESET}\n")
        
        print(f"{BLUE}GMM with k = {params["k_gmm1"]}{RESET}\n")
        trained_gmm_model = run_gmm_model(params["k_gmm1"], train_data, seed=params["seed"])
    except ValueError:
        print(f"{RED}My GMM Class did not work, so now using sklearn GMM class{RESET}")

        print("Running my GMM class for arbitrary value of k")
        gmm_model = GaussianMixture(10)
        gmm_model.fit(train_data)
        gmm_model.predict(train_data)
        overall_log_likelihood = gmm_model.score(train_data) * train_data.shape[0]

        print(f"{GREEN}sklearn GMM Class worked{RESET}")
        print("Plotting AIC BIC graph using sklearn GMM Class\n")

        gmm_aic_bic_method(params["max_k"], train_data, save_as="aic_bic_plot_original_dataset", seed=params["seed"], model_type="sklearn")

        # Lower values of AIC and BIC are better
        print(f"{MAGENTA}Inferred value of k_gmm1 = {params["k_gmm1"]}{RESET}\n")

        print(f"{BLUE}GMM with k = {params["k_gmm1"]}{RESET}\n")
        trained_sklearn_gmm_model = run_gmm_model(params["k_gmm1"], train_data, seed=params["seed"], model_type="sklearn")

    # =================================================================
    #                         5.1, 5.2 & 5.3   
    # =================================================================
    pca(original_data, train_data, save=True)

    print(f"{MAGENTA}Inferred value of k2 = {params["k2"]}{RESET}\n")

    # =================================================================
    #          6.1 & 6.2 - Scree plot, reduced dataset, k_kmeans3
    # =================================================================
    print(f"{BLUE}KMeans with k = {params["k2"]}{RESET}\n")
    run_k_means_model(params["k2"], train_data, seed=params["seed"], init_method=params["init_method"])

    # Scree Plot
    draw_scree_plot(train_data, save_as="pca/scree_plot_full_dataset.png")

    # Reducing the dataset to the optimal dimensions
    print(f"{MAGENTA}Inferred value of op_dim = {params["op_dim"]}{RESET}\n")

    pca_model = PCA(params["op_dim"])
    pca_model.load_data(train_data)
    pca_model.fit()
    reduced_dataset = pca_model.transform()
    print(f"{GREEN}Dataset reduced to {params["op_dim"]} dimensions{RESET}")

    # Elbow plot for reduced dataset to find optimal number of clusters from reduced dataset
    draw_elbow_plot(reduced_dataset, save_as="elbow_point_reduced_dataset", max_k=params["max_k"], init_method=params["init_method"])

    print(f"{MAGENTA}Inferred value of k_kmeans3 = {params["k_kmeans3"]}{RESET}\n")
    print(f"{BLUE}KMeans with k = {params["k_kmeans3"]}{RESET}\n")
    run_k_means_model(params["k_kmeans3"], reduced_dataset, seed=params["seed"], init_method=params["init_method"])

    print(f"{BLUE}GMM using k = {params["k2"]}{RESET}\n")
    # run_gmm_model(params["k2"], train_data, seed=params["seed"], model_type="my")
    run_gmm_model(params["k2"], train_data, seed=params["seed"], model_type="sklearn")

    gmm_aic_bic_method(params["max_k"], reduced_dataset, save_as="aic_bic_plot_reduced_dataset", seed=params["seed"], model_type="sklearn")
    print(f"{MAGENTA}Inferred value of k_gmm3 = {params["k_gmm3"]}{RESET}\n")

    print(f"{BLUE}GMM with k = {params["k_gmm3"]}{RESET}\n")
    # run_gmm_model(params["k_gmm3"], reduced_dataset, seed=params["seed"], model_type="my")
    run_gmm_model(params["k_gmm3"], reduced_dataset, seed=params["seed"], model_type="sklearn")

    # =================================================================
    #               7.1, 7.2 & 7.3 - Cluster Analysis
    # =================================================================
    k_means_cluster_analysis(params, train_data, original_data)
    # Looking at the silhouette score and inertia score we identify the value of k_kmeans (we want more silhouette score and less inertia score)
    print(f"{MAGENTA}Inferred value of k_kmeans = {params["k_kmeans"]}{RESET}\n")
    gmm_cluster_analysis(params, train_data, original_data, model_type="sklearn")
    print(f"{MAGENTA}Inferred value of k_gmm = {params["k_gmm"]}{RESET}\n")
    compare_kmeans_gmm(params, train_data, original_data)

    # =================================================================
    #                  8 - Hierarchical Clustering
    # =================================================================
    hierarchical_clustering(params, original_data, train_data)

    # =================================================================
    #                  9.1 & 9.2 - Spotify Dataset
    # =================================================================
    # spotify_dataset()
