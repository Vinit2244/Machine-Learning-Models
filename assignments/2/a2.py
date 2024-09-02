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
RESET = "\033[0m"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM
from models.pca.pca import PCA

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
def run_k_means_model(k, train_data, test_data, val_data, seed=None):
    k_means_model = KMeans(k)
    k_means_model.load_data(train_data)
    start_time = time.time()
    epochs_taken = k_means_model.fit(seed)
    end_time = time.time()
    print(f"\t{GREEN}Time taken to fit: {round(end_time - start_time, 5)} s{RESET}")
    print(f"\t{GREEN}Epochs taken to fit: {epochs_taken}{RESET}")
    # k_means_model.predict(test_data)
    final_cost = round(k_means_model.getCost(), 5)
    print(f"\t{GREEN}Final Cost: {final_cost}{RESET}")
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")
    return final_cost

# Implements the KMeans algorithm for 3.1 and 3.2
def k_means(k, data):
    # # 3.1
    train_data, test_data, val_data = split_data(data, 100, 0, shuffle=False)
    print(f"{BLUE}For 3.1{RESET}\n")
    run_k_means_model(k, train_data, test_data, val_data, seed=None)

    # 3.2
    '''
        Source: https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/

        Not always the best method: The elbow method might not be suitable for all datasets, 
        especially for those with high dimensionality or clusters of irregular shapes.
        Hence if you see the graph it is very difficult to identify the elbow point.
    '''
    # Decided upon the value of 16 after watching the graph till various points (max = 200) (no. of data points in data)
    k_arr = range(1, 15 + 1)
    final_cost_arr = []
    print(f"{BLUE}For 3.2{RESET}\n")
    for k in k_arr:
        print(f"{BLUE}k = {k}{RESET}")
        final_cost = run_k_means_model(k, train_data, test_data, val_data, seed=None)
        final_cost_arr.append(final_cost)

    # Plotting k vs final cost
    plt.figure(figsize=(10, 6))
    plt.plot(k_arr, final_cost_arr, 'o-')
    plt.xlabel('k')
    plt.ylabel('Final WCSS Cost')
    # plt.ylim(bottom=0)
    plt.title('k vs Final WCSS Cost')
    plt.savefig('./assignments/2/figures/k_means/k_vs_final_cost.png')
    print(f"\n{GREEN}k vs Final WCSS Cost plot saved{RESET}\n")
    plt.close()

    '''
        After inspecting the graph I determined the elbow point at k = 5
    '''
    k_kmeans1 = 5 # Or we can keep 11 as well, check with other ppl when they are done with this part
    print(f"{BLUE}For k_kmeans1{RESET}\n")
    print(f"{BLUE}For 3.2 with k = {k_kmeans1}{RESET}\n")
    run_k_means_model(k_kmeans1, train_data, test_data, val_data, seed=None)

# Implements the GMM algorithm for 4.1 and 4.2
def gmm(k, data):
    gmm_model = GMM(k)
    gmm_model.load_data(data)
    pass

# Implements the PCA algorithm for 5.1, 5.2 and 5.3
def pca(n_components, data):
    pca_model = PCA(n_components)

    # Note that to make the PCA model generalised it only takes input as 2D numpy array
    data = data.tolist()
    pca_data = []
    for row in data:
        pca_data.append(row[1].tolist())
    pca_data = np.array(pca_data)
    pca_model.load_data(pca_data)
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
        plt.show()
        plt.savefig('./assignments/2/figures/pca/pca_3.png')
        print(f"{GREEN}PCA plot saved{RESET}\n")
        plt.close()

if __name__ == "__main__":
    # ------------------------- Reading Data --------------------------
    data_path = './data/external/word-embeddings.feather' # Path to the word embeddings file
    df = pd.read_feather(data_path)                       # First column is the word, second column is the embedding (as a numpy array) of 512 length
    data = df.to_numpy()
    print(data[:,0])
    # --------------------------- 3.1 & 3.2 ---------------------------
    # n_clusters = 10
    # k_means(n_clusters, data)

    # --------------------------- 4.1 & 4.2 ---------------------------
    # gmm(n_clusters, data)

    # --------------------------- 5.1, 5.2 & 5.3 ---------------------------
    # n_components = 2
    # pca(n_components, data)

    # n_components = 3
    # pca(n_components, data)