# pip3 install pandas pyarrow          # For reading the .feather file
import pandas as pd
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    "data_path":     "./data/external/temp_data_k_means.csv",
    "train_percent": 100,       # Unsupervised learning
    "test_percent":  0,
    "val_percent":   0,
    "shuffle":       False,
    "seed":          None,
    "max_k":         15,        # Decided upon the value of 15 after watching the graph till various points (max = 200) (no. of data points in data)
    "init_method":   "kmeans",  # Can be kmeans or kmeans_pp
    "k_kmeans1":     3,         # Based on visual inspection of the k vs MCSS plot deciding the number of clusters (elbow point)
    "n_clusters":    10,        # Arbitrary value
    "k2":            3,         # Based on visual inspection of the PCA plots deciding the number of clusters
    "op_dim":        2,         # Based on visual inspection of the scree plot deciding the number of optimal dimensions
    "k_kmeans3":     3,         # Based on visual inspection of the k vs MCSS plot on reduced dataset deciding the number of clusters (elbow point)
}

original_data_params = {
    "data_path":     "./data/external/word-embeddings.feather",
    "train_percent": 100,       # Unsupervised learning
    "test_percent":  0,
    "val_percent":   0,
    "shuffle":       False,
    "seed":          None,
    "max_k":         15,        # Decided upon the value of 15 after watching the graph till various points (max = 200) (no. of data points in data)
    "init_method":   "kmeans",  # Can be kmeans or kmeans_pp
    "k_kmeans1":     5,         # Based on visual inspection of the k vs MCSS plot deciding the number of clusters (elbow point)
    "n_clusters":    10,        # Arbitrary value
    "k2":            5,         # Based on visual inspection of the PCA plots deciding the number of clusters
    "op_dim":        4,         # Based on visual inspection of the scree plot deciding the number of optimal dimensions
    "k_kmeans3":     6,         # Based on visual inspection of the k vs MCSS plot on reduced dataset deciding the number of clusters (elbow point)
}

# Reads the data from the given path and returns it as a numpy array after processing
def read_data(data_path):
    # First column is the word, second column is the embedding (as a numpy array) of 512 length
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        data = df.to_numpy()
        processed_data = []
        for row in data:
            processed_data.append(row.tolist())
        processed_data = np.array(processed_data)
    elif data_path.endswith('.feather'):
        df = pd.read_feather(data_path)
        data = df.to_numpy()
        processed_data = []
        for row in data:
            processed_data.append(row[1].tolist())
        processed_data = np.array(processed_data)
    return processed_data

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
def run_k_means_model(k, train_data, seed=None, init_method="kmeans"):
    k_means_model = KMeans(k, init_method=init_method)
    k_means_model.load_data(train_data)
    start_time = time.time()
    epochs_taken = k_means_model.fit(seed)
    end_time = time.time()
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

# Implements the GMM algorithm for 4.1 and 4.2
def gmm(k, data):
    gmm_model = GMM(k)
    gmm_model.load_data(data)

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

def draw_scree_plot(train_data):
    pca_model = PCA()
    pca_model.load_data(train_data)
    eigenvalues, eigenvectors = pca_model.get_eigenvalues_eigenvectors()
    max_val = 21
    x_axis = range(1, len(eigenvalues) + 1)
    plt.plot(x_axis[:max_val], eigenvalues[:max_val], 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.savefig('./assignments/2/figures/pca/scree_plot.png')
    print(f"{GREEN}Scree plot saved{RESET}\n")

if __name__ == "__main__":
    params = original_data_params

    data = read_data(params["data_path"])
    train_data, test_data, val_data = split_data(data, params["train_percent"], params["test_percent"], shuffle=params["shuffle"])

    # --------------------------- 3.1 & 3.2 ---------------------------
    k_means_elbow_method(train_data, save_as="k_vs_final_cost_3.2", max_k=params["max_k"], init_method=params["init_method"])
    print(f"{MAGENTA}k_kmeans1 = {params["k_kmeans1"]}{RESET}\n")
    print(f"{BLUE}For 3.2 with k = {params["k_kmeans1"]}{RESET}")
    run_k_means_model(params["k_kmeans1"], train_data, seed=params["seed"], init_method=params["init_method"])

    # --------------------------- 4.1 & 4.2 ---------------------------
    # n_clusters = params["n_clusters"]
    # gmm(n_clusters, train_data)

    # ------------------------- 5.1, 5.2 & 5.3 ------------------------
    n_components = 2
    pca(n_components, train_data)

    n_components = 3
    pca(n_components, train_data)

    # 6.1 - Performing KMeans clustering with k = k2
    print(f"{MAGENTA}k2 = {params["k2"]}{RESET}\n")
    print(f"{BLUE}For 6.1 with k = {params["k2"]}{RESET}\n")
    run_k_means_model(params["k2"], train_data, seed=params["seed"], init_method=params["init_method"])

    # 6.2 - Scree plot, reduced dataset k_kmeans3, clustering with k = k_kmeans3
    print(f"{BLUE}For 6.2{RESET}\n")
    # Scree Plot
    draw_scree_plot(train_data)

    # Reducing the dataset to the optimal dimensions
    print(f"{MAGENTA}op_dim = {params["op_dim"]}{RESET}\n")
    pca_model = PCA(params["op_dim"])
    pca_model.load_data(train_data)
    pca_model.fit()
    reduced_dataset = pca_model.transform()
    print(f"{GREEN}Dataset reduced to {params["op_dim"]} dimensions{RESET}")

    # Finding the optimal number of clusters for the reduced dataset
    k_means_elbow_method(reduced_dataset, save_as="k_vs_final_cost_6.2", max_k=params["max_k"], init_method=params["init_method"])
    print(f"{MAGENTA}k_kmeans3 = {params["k_kmeans3"]}{RESET}\n")
    print(f"{BLUE}For 6.2 with k = {params["k_kmeans3"]}{RESET}\n")
    run_k_means_model(params["k_kmeans3"], reduced_dataset, seed=params["seed"], init_method=params["init_method"])