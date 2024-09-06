# pip3 install pandas pyarrow          # For reading the .feather file
import pandas as pd
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as hc

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

# Data params dictionaries contain all the parameters that are set manually through inspection
test_data_params = {
    "type":          "test data",
    "data_path":     "./data/external/temp_data_k_means.csv",
    "train_percent": 100,       # Unsupervised learning
    "test_percent":  0,
    "val_percent":   0,
    "shuffle":       False,
    "seed":          None,
    "max_k":         15,        # Decided upon the value of 15 after watching the graph till various points (max = 200) (no. of data points in data)
    "init_method":   "kmeans",  # Can be kmeans or kmeans_pp
    "k_kmeans1":     3,         # Based on visual inspection of the k vs MCSS plot deciding the number of clusters (elbow point)
    "k2":            3,         # Based on visual inspection of the PCA plots deciding the number of clusters
    "op_dim":        2,         # Based on visual inspection of the scree plot deciding the number of optimal dimensions
    "k_kmeans3":     3,         # Based on visual inspection of the k vs MCSS plot on reduced dataset deciding the number of clusters (elbow point)
    "label_idx":     2,         # Column Index of the label in the data
    "k_kmeans":      3,         # Based on interpreting the clustered labels deciding the number of clusters
    "linkage":       "ward",    # Based on interpreting the dendrograms deciding the best linkage method
    "k_best1":       3,
    "k_best2":       3,
}

original_data_params = {
    "type":          "original assignment data",
    "data_path":     "./data/external/word-embeddings.feather",
    "train_percent": 100,       # Unsupervised learning
    "test_percent":  0,
    "val_percent":   0,
    "shuffle":       False,
    "seed":          None,
    "max_k":         15,        # Decided upon the value of 15 after watching the graph till various points (max = 200) (no. of data points in data)
    "init_method":   "kmeans",  # Can be kmeans or kmeans_pp
    "k_kmeans1":     5,         # Based on visual inspection of the k vs MCSS plot deciding the number of clusters (elbow point)
    "k2":            5,         # Based on visual inspection of the PCA plots deciding the number of clusters
    "op_dim":        4,         # Based on visual inspection of the scree plot deciding the number of optimal dimensions
    "k_kmeans3":     6,         # Based on visual inspection of the k vs MCSS plot on reduced dataset deciding the number of clusters (elbow point)
    "label_idx":     0,         # Column Index of the label in the data
    "k_kmeans":      5,         # Based on interpreting the clustered labels deciding the number of clusters
    "linkage":       "ward",    # Based on interpreting the dendrograms deciding the best linkage method
    "k_best1":       5,
    "k_best2":       5,
}

# Reads the data from the given path and returns it as a numpy array after processing
def read_data(data_path):
    # First column is the word, second column is the embedding (as a numpy array) of 512 length
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        original_data = df.to_numpy()
        processed_data = []
        for row in original_data:
            processed_data.append(row[:-1:].tolist())
        processed_data = np.array(processed_data)
    elif data_path.endswith('.feather'):
        df = pd.read_feather(data_path)
        original_data = df.to_numpy()
        processed_data = []
        for row in original_data:
            processed_data.append(row[1].tolist())
        processed_data = np.array(processed_data)
    return original_data, processed_data

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

# Runs a KMeans model with the given inputs and prints statistics
def run_k_means_model(k, train_data, seed=None, init_method="kmeans", print_op=True):
    k_means_model = KMeans(k, init_method=init_method)
    k_means_model.load_data(train_data)
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

# Implements the KMeans algorithm for 3.1 and 3.2
def k_means_elbow_method(train_data, save_as, max_k=15, init_method="kmeans"):
    # 3.2
    '''
        Source: https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/

        Not always the best method: The elbow method might not be suitable for all datasets, 
        especially for those with high dimensionality or clusters of irregular shapes.
        Hence if you see the graph it is very difficult to identify the elbow point.
    '''
    k_arr = range(1, max_k + 1)
    final_cost_arr = []
    for k in k_arr:
        print(f"{BLUE}k = {k}{RESET}")
        trained_k_means_model = run_k_means_model(k, train_data, seed=None, init_method=init_method)
        final_cost_arr.append(round(trained_k_means_model.getCost(), 5))

    # Plotting k vs final cost
    plt.figure(figsize=(10, 6))
    plt.plot(k_arr, final_cost_arr, 'o-')
    plt.xlabel('k')
    plt.ylabel('Final WCSS Cost')
    # plt.ylim(bottom=0)
    plt.title('k vs Final WCSS Cost')
    plt.savefig(f'./assignments/2/figures/k_means/{save_as}.png')
    print(f"{GREEN}{save_as} plot saved{RESET}\n")
    plt.close()

# Runs a GMM model with the given inputs and prints statistics
def run_gmm_model(k, train_data):
    gmm_model = GMM(k)
    gmm_model.load_data(train_data)
    start_time = time.time()
    epochs_taken = gmm_model.fit()
    end_time = time.time()
    print(f"\t{GREEN}Time taken to fit: {round(end_time - start_time, 5)} s{RESET}")
    print(f"\t{GREEN}Epochs taken to fit: {epochs_taken}{RESET}")
    overall_likelihood = round(gmm_model.getLikelihood(), 5)
    print(f"\t{GREEN}Final Likelihood: {overall_likelihood}{RESET}")
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")
    return gmm_model

# Implements the GMM algorithm for 4.1 and 4.2
def gmm_aic_bic_method(max_k, train_data, save_as):
    # 4.2
    '''
        Source: https://www.youtube.com/watch?v=4al2LfJz6Q8
    '''
    k_arr = range(1, max_k + 1)
    aic_arr = []
    bic_arr = []
    for k in k_arr:
        print(f"{BLUE}k = {k}{RESET}")
        trained_gmm_model = run_gmm_model(k, train_data)
        overall_likelihood = round(trained_gmm_model.getLikelihood(), 5)
        aic = (2 * k) - (2 * np.log(overall_likelihood + 1e-10))
        bic = (k * np.log(train_data.shape[0] + 1e-10)) - (2 * np.log(overall_likelihood + 1e-10))
        aic_arr.append(round(aic, 5))
        bic_arr.append(round(bic, 5))

    # Plotting k vs aic and bic
    plt.figure(figsize=(10, 6))
    plt.plot(k_arr, aic_arr, 'o-', label='AIC', color='red')
    plt.plot(k_arr, bic_arr, 'o-', label='BIC', color='blue')
    plt.xlabel('k')
    plt.ylabel('AIC/BIC')
    # plt.ylim(bottom=0)
    plt.title('k vs AIC/BIC')
    plt.savefig(f'./assignments/2/figures/gmm/{save_as}.png')
    print(f"{GREEN}{save_as} plot saved{RESET}\n")
    plt.close()

# Implements the PCA algorithm for 5.1, 5.2 and 5.3
def pca(n_components, data):
    pca_model = PCA(n_components)
    pca_model.load_data(data)
    pca_model.fit()
    transformed_data = pca_model.transform()
    if pca_model.checkPCA():
        print(f"{GREEN}PCA transformation for {n_components} components successful{RESET}")
    else:
        print(f"{RED}PCA transformation for {n_components} unsuccessful{RESET}")
        return
    if n_components == 2:
        # Plotting the transformed data
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Principal Component Analysis')
        # plt.show()
        plt.savefig('./assignments/2/figures/pca/pca_2.png')
        print(f"{GREEN}PCA plot saved{RESET}\n")
        plt.close()
    elif n_components == 3:
        # Plotting the transformed data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title('Principal Component Analysis')
        # plt.show()
        plt.savefig('./assignments/2/figures/pca/pca_3.png')
        print(f"{GREEN}PCA plot saved{RESET}\n")
        plt.close()

# Plots scree plot for deciding optimal number of dimensions for PCA
def draw_scree_plot(train_data, save_as):
    pca_model = PCA()
    pca_model.load_data(train_data)
    eigenvalues, eigenvectors = pca_model.get_eigenvalues_eigenvectors()
    max_val = 21
    x_axis = range(1, len(eigenvalues) + 1)
    plt.plot(x_axis[:max_val], eigenvalues[:max_val], 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.savefig(f'./assignments/2/figures/{save_as}')
    print(f"{GREEN}Scree plot saved{RESET}\n")

# Given the cluster assignment, original data and the label index (index of column which represents label), returns the clusters of labels
def get_cluster_values(k, cluster_assignment, original_data, label_idx):
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
def plot_dendrogram(type, original_data, preprocessed_data, method, save_as):
    Z = hc.linkage(preprocessed_data, method=method, metric="euclidean")
    if type == "original assignment data":
        custom_labels = original_data[:, 0].tolist()
    elif type == "test data":
        custom_labels = original_data[:, 2].tolist()
    fig = plt.figure(figsize=(25, 10))
    dn = hc.dendrogram(Z, labels=custom_labels)
    plt.savefig(f'./assignments/2/figures/hierarchical/{save_as}.png')
    print(f"{GREEN}{save_as} plot saved{RESET}\n")

# For preprocessing the spotify dataset
def preprocess_spotify_dataset(data, preprocessed_data_file_path):
    # Renaming the first unnamed column as index
    data = data.rename(columns={'Unnamed: 0': 'index'})

    # Remove exact duplicates - if track id is same, the song is exactly the same
    data = data.drop_duplicates("track_id")

    # Remove duplicate songs in multiple albums
    data = data.drop_duplicates("track_name")

    # Removing unnecessary columns and non-numeric columns
    unnecessary_cols = ["index", "track_id", "album_name", "artists", "track_name", "explicit", "track_genre"]
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

if __name__ == "__main__":
    params = original_data_params

    original_data, preprocessed_data = read_data(params["data_path"])
    train_data, test_data, val_data = split_data(preprocessed_data, params["train_percent"], params["test_percent"], shuffle=params["shuffle"])

    # # --------------------------- 3.1 & 3.2 ---------------------------
    # k_means_elbow_method(train_data, save_as="k_vs_final_cost_3.2", max_k=params["max_k"], init_method=params["init_method"])
    # print(f"{MAGENTA}k_kmeans1 = {params["k_kmeans1"]}{RESET}\n")
    # print(f"{BLUE}For 3.2 with k = {params["k_kmeans1"]}{RESET}")
    # run_k_means_model(params["k_kmeans1"], train_data, seed=params["seed"], init_method=params["init_method"])

    # # # --------------------------- 4.1 & 4.2 ---------------------------
    # # gmm_aic_bic_method(params["max_k"], train_data, save_as="k_vs_aic_bic_4.2")

    # # ------------------------- 5.1, 5.2 & 5.3 ------------------------
    # n_components = 2
    # pca(n_components, train_data)

    # n_components = 3
    # pca(n_components, train_data)

    # # ------------------------- 6.1 - Performing KMeans clustering with k = k2 -------------------------
    # print(f"{MAGENTA}k2 = {params["k2"]}{RESET}\n")
    # print(f"{BLUE}For 6.1 with k = {params["k2"]}{RESET}\n")
    # run_k_means_model(params["k2"], train_data, seed=params["seed"], init_method=params["init_method"])

    # # ------------------------- 6.2 - Scree plot, reduced dataset k_kmeans3, clustering with k = k_kmeans3 -------------------------
    # print(f"{BLUE}For 6.2{RESET}\n")
    # # Scree Plot
    # draw_scree_plot(train_data, save_as="pca/scree_plot_6_2.png")

    # # Reducing the dataset to the optimal dimensions
    # print(f"{MAGENTA}op_dim = {params["op_dim"]}{RESET}\n")
    # pca_model = PCA(params["op_dim"])
    # pca_model.load_data(train_data)
    # pca_model.fit()
    # reduced_dataset = pca_model.transform()
    # print(f"{GREEN}Dataset reduced to {params["op_dim"]} dimensions{RESET}")

    # # Finding the optimal number of clusters for the reduced dataset
    # k_means_elbow_method(reduced_dataset, save_as="k_vs_final_cost_6.2", max_k=params["max_k"], init_method=params["init_method"])
    # print(f"{MAGENTA}k_kmeans3 = {params["k_kmeans3"]}{RESET}\n")
    # print(f"{BLUE}For 6.2 with k = {params["k_kmeans3"]}{RESET}\n")
    # run_k_means_model(params["k_kmeans3"], reduced_dataset, seed=params["seed"], init_method=params["init_method"])

    # # ------------------------- 7.1 - K-means cluster analysis -------------------------
    # trained_model_k_kmeans1 = run_k_means_model(params["k_kmeans1"], train_data, print_op=False)
    # clustered_values_k_kmeans1 = get_cluster_values(params["k_kmeans1"], trained_model_k_kmeans1.cluster_labels, original_data, params["label_idx"])
    # print(f"{BLUE}Clustered labels for k = {params["k_kmeans1"]}{RESET}")
    # for cluster_idx, cluster in enumerate(clustered_values_k_kmeans1):
    #     print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
    #     print(f"{cluster}\n")

    # print("------------------------------------------------------------------------------------------------------")

    # trained_model_k2 = run_k_means_model(params["k2"], train_data, print_op=False)
    # clustered_values_k2 = get_cluster_values(params["k2"], trained_model_k2.cluster_labels, original_data, params["label_idx"])
    # print(f"{BLUE}Clustered labels for k = {params["k2"]}{RESET}")
    # for cluster_idx, cluster in enumerate(clustered_values_k2):
    #     print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
    #     print(f"{cluster}\n")

    # print("------------------------------------------------------------------------------------------------------")

    # trained_model_k_kmeans3 = run_k_means_model(params["k_kmeans3"], train_data, print_op=False)
    # clustered_values_k_kmeans3 = get_cluster_values(params["k_kmeans3"], trained_model_k_kmeans3.cluster_labels, original_data, params["label_idx"])
    # print(f"{BLUE}Clustered labels for k = {params["k_kmeans3"]}{RESET}")
    # for cluster_idx, cluster in enumerate(clustered_values_k_kmeans3):
    #     print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
    #     print(f"{cluster}\n")

    # x_axis = [f"k_kmeans1 = {params["k_kmeans1"]}", f"k2 = {params["k2"]}", f"k_kmeans3 = {params["k_kmeans3"]}"]
    # y_axis = [trained_model_k_kmeans1.getCost(), trained_model_k2.getCost(), trained_model_k_kmeans3.getCost()]
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_axis, y_axis)
    # plt.xlabel('k')
    # plt.ylabel('Final WCSS Cost')
    # plt.title('k vs Final WCSS Cost')
    # plt.savefig('./assignments/2/figures/k_means/k_vs_final_cost_7.1.png')
    # print(f"{GREEN}k vs Final WCSS Cost plot saved{RESET}\n")

    # # On interpreting the clusters we identify the value of k_kmeans
    # print(f"{MAGENTA}k_kmeans = {params["k_kmeans"]}{RESET}\n")

    # # ------------------------- 8 - Hierarchical Clustering ------------------------
    # # Single linkage: based on the minimum distance between points in the clusters
    # plot_dendrogram(params["type"], original_data, train_data, method='single', save_as='dendrogram_single')

    # # Complete linkage: based on the maximum distance between points in the clusters
    # plot_dendrogram(params["type"], original_data, train_data, method='complete', save_as='dendrogram_complete')

    # # Average linkage: based on the average distance between all points in the clusters
    # plot_dendrogram(params["type"], original_data, train_data, method='average', save_as='dendrogram_average')

    # # Ward method: minimizes the total variance within clusters (sum of squared differences)
    # plot_dendrogram(params["type"], original_data, train_data, method='ward', save_as='dendrogram_ward')

    # # Centroid linkage: based on the distance between the centroids (means) of the clusters
    # plot_dendrogram(params["type"], original_data, train_data, method='centroid', save_as='dendrogram_centroid')

    # # Median linkage: similar to centroid, but uses the median rather than the mean
    # plot_dendrogram(params["type"], original_data, train_data, method='median', save_as='dendrogram_median')

    # # On interpreting the dendrograms we identify the best linkage method
    # print(f"{MAGENTA}Linkage = {params["linkage"]}{RESET}\n")

    # Z = hc.linkage(train_data, method=params["linkage"], metric='euclidean')
    # # For kbest1 (K-Means)
    # clusters_kbest1 = hc.fcluster(Z, t=params["k_best1"], criterion='maxclust')

    # # For kbest2 (GMM)
    # clusters_kbest2 = hc.fcluster(Z, t=params["k_best2"], criterion='maxclust')

    # clustered_values_kbest1 = get_cluster_values(params["k_best1"], clusters_kbest1, original_data, params["label_idx"])
    # clustered_values_kbest2 = get_cluster_values(params["k_best2"], clusters_kbest2, original_data, params["label_idx"])
    
    # print(f"{BLUE}Clustered labels for k_best1 = {params["k_best1"]}{RESET}")
    # for cluster_idx, cluster in enumerate(clustered_values_kbest1):
    #     print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
    #     print(f"{cluster}\n")

    # print(f"{BLUE}Clustered labels for k_best2 = {params["k_best2"]}{RESET}")
    # for cluster_idx, cluster in enumerate(clustered_values_kbest1):
    #     print(f"\n{GREEN}Cluster {cluster_idx + 1}{RESET}:")
    #     print(f"{cluster}\n")

    # ------------------------- 9 - Spotify Dataset ------------------------
    # ------------------------- 9.1 - PCA + KNN ------------------------
    # Dataset 1 - spotify.csv
    spotify_data_file_path = './data/external/spotify.csv' 
    spotify_preprocessed_data_file_path = './data/interim/2/preprocessed_spotify.csv'
    # Some data points have commas in them, so we need to use quotechar to read the file
    # Source: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    try:
        # Checking if preprocessed data is already present
        spotify_data = pd.read_csv(spotify_preprocessed_data_file_path, quotechar='"')
    except FileNotFoundError:
        spotify_data = pd.read_csv(spotify_data_file_path, quotechar='"')
        spotify_data = preprocess_spotify_dataset(spotify_data, spotify_preprocessed_data_file_path)
    headers = list(spotify_data.columns)
    data = spotify_data.to_numpy()
    draw_scree_plot(data, save_as="spotify_dataset/scree_plot_9_1.png")
