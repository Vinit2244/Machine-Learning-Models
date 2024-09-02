import numpy as np
import random

class KMeans:
    def __init__(self, k):
        self.k = k          # Number of clusters
        self.data = None
        self.cluster_labels = None
        self.cluster_centers = None
    
    # Load all the data
    def load_data(self, data):
        self.data = data    # Data to cluster
        self.cluster_labels = np.full(data.shape[0], -1)
        self.cluster_centers = np.zeros((self.k, self.data[0][1].shape[0]))

    # Calculate the distance () between a single data point and an array of data points
    def calc_distance(self, x1, x2_arr):
        return np.linalg.norm(x2_arr - x1, axis=1)
    
    # Initialise random k cluster centers
    def initialise_clusters(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Selecting k random index between 0 and len(data)
        indexes = random.sample(range(0, self.data.shape[0]), self.k)

        # Assigning cluster center labels and cluster center values
        for cluster_number, index in enumerate(indexes):
            self.cluster_labels[index] = cluster_number
            self.cluster_centers[cluster_number] = self.data[index][1]

    # Assigns each data point to a cluster until convergence
    def fit(self, seed=None):
        self.initialise_clusters(seed)

        n_epochs = 0
        while True:
            did_clusters_change = False

            # Expectation Step
            for data_idx, data_point in enumerate(self.data):
                distances = self.calc_distance(data_point[1], self.cluster_centers)
                # Label of cluster is the same as it's index
                nearest_cluster_label = np.argmin(distances) 
                if self.cluster_labels[data_idx] != nearest_cluster_label:
                    # If the new label is different from the old label, then the clusters have changed
                    did_clusters_change = True
                    self.cluster_labels[data_idx] = nearest_cluster_label
            
            # Maximisation Step
            for cluster_label in range(self.k):
                cluster_data = self.data[self.cluster_labels == cluster_label]
                if len(cluster_data) == 0:
                    continue
                if cluster_data.shape[0] == 1:
                    self.cluster_centers[cluster_label] = cluster_data[0][1]
                else:
                    self.cluster_centers[cluster_label] = np.mean(cluster_data[:, 1], axis=0)
            
            n_epochs += 1

            if not did_clusters_change:
                break
            
        return n_epochs

    # Predicts the cluster for each data point
    def predict(self, data_to_predict):
        predicted_clusters = np.full(data_to_predict.shape[0], -1)
        for data_idx, data_point in enumerate(data_to_predict):
            distances = self.calc_distance(data_point[1], self.cluster_centers)
            nearest_cluster_label = np.argmin(distances)
            predicted_clusters[data_idx] = nearest_cluster_label
        
        return predicted_clusters

    # Returns WCSS cost of the model
    def getCost(self):
        cost = 0
        for data_idx, data_point in enumerate(self.data):
            cluster_label = self.cluster_labels[data_idx]
            cluster_center = self.cluster_centers[cluster_label]
            cost += np.linalg.norm(data_point[1] - cluster_center) ** 2
        return cost
