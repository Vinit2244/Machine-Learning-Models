# pip3 install pandas pyarrow          # For reading the .feather file
import pandas as pd
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Colors for printing for better readability
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.k_means.k_means import KMeans

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
    # Decided upon the value of 31 after watching the graph till 200 (no. of data points in data)
    k_arr = range(1, 31)
    final_cost_arr = []
    print(f"{BLUE}For 3.2{RESET}\n")
    print(f"{BLUE}k = {k}{RESET}")
    for k in k_arr:
        run_k_means_model(k, train_data, test_data, val_data, seed=None)

    # Plotting k vs final cost
    plt.plot(k_arr, final_cost_arr, '.-')
    plt.xlabel('k')
    plt.ylabel('Final WCSS Cost')
    plt.title('k vs Final WCSS Cost')
    plt.savefig('./assignments/2/figures/k_means/k_vs_final_cost.png')
    print(f"\n{GREEN}k vs Final WCSS Cost plot saved{RESET}\n")
    plt.close()

    '''
        After inspecting the graph I determined the elbow point at k = 12 or 13
    '''
    k_kmeans1 = 13
    print(f"{BLUE}For k_kmeans1{RESET}\n")
    print(f"{BLUE}For 3.2 with k = {k_kmeans1}{RESET}\n")
    run_k_means_model(k_kmeans1, train_data, test_data, val_data, seed=None)

if __name__ == "__main__":
    # ------------------------- Reading Data --------------------------
    data_path = './data/external/word-embeddings.feather' # Path to the word embeddings file
    df = pd.read_feather(data_path)                       # First column is the word, second column is the embedding (as a numpy array) of 512 length
    data = df.to_numpy()

    # --------------------------- 3.1 & 3.2 ---------------------------
    n_clusters = 10
    k_means(n_clusters, data)