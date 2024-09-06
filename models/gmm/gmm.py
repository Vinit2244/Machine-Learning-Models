import numpy as np
import random
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k):
        self.k = k          # Number of gaussian models
        self.data = None    # 2D numpy array
        self.weights = None
        self.means = None
        self.cov_matrices = None
        self.responsibility_mat = None
        self.arr_log_likelihood = None

    def load_data(self, data):
        self.data = data

    def initialise_params(self):
        # Initialise weights
        weights = np.random.rand(self.k)         # Initialise some random values
        self.weights = weights / np.sum(weights) # Normalise the weights

        # Initialise means
        indexes = random.sample(range(0, self.data.shape[0]), self.k)
        self.means = self.data[indexes]

        # Initialise variances
        # if no of samples < dimensions, then use diagonal matrix (each feature independent of other features)
        if self.data.shape[0] < self.data.shape[1]:
            self.cov_matrices = [np.diag(np.random.rand(self.data.shape[1])) for _ in range(self.k)]
            self.cov_matrices = self.cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6  # Regularization to avoid singularity
        # if no of samples > dimensions, then use full covariance matrix
        else:
            self.cov_matrices = []
            for _ in range(self.k):
                A = np.random.rand(self.data.shape[1], self.data.shape[1])
                cov_matrix = np.dot(A, A.T)
                cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6  # Regularization to avoid singularity
                self.cov_matrices.append(cov_matrix)

    # EM algorithm
    def fit(self):
        self.initialise_params()

        self.arr_log_likelihood = [0]

        epochs_taken = 0

        while True:
            # Expectation step
            responsibility_mat = np.zeros((self.data.shape[0], self.k))
            for data_point_idx, data_point in enumerate(self.data):
                for cluster_idx in range(self.k):
                    weight, mean, cov_matrix = self.weights[cluster_idx], self.means[cluster_idx], self.cov_matrices[cluster_idx]
                    # Calculate the responsibility of each cluster for a data point
                    responsibility_mat[data_point_idx][cluster_idx] = weight * multivariate_normal.pdf(data_point, mean, cov_matrix, allow_singular=True)
                # Normalising the responsibility for a data point for each cluster
                responsibility_mat[data_point_idx] /= np.sum(responsibility_mat[data_point_idx]) + 1e-10

            # Maximization step
            # Total responsibility assigned to each cluster
            total_responsibility_per_cluster = np.sum(responsibility_mat, axis=0)

            # Update weights
            total_responsibility = np.sum(total_responsibility_per_cluster)
            responsibility_per_cluster = total_responsibility_per_cluster / total_responsibility
            self.weights = responsibility_per_cluster

            # Update means
            for cluster_idx in range(self.k):
                # Weighted sum of all data points assigned to a cluster
                weighted_sum = np.zeros(self.data.shape[1])
                for data_point_idx, data_point in enumerate(self.data):
                    weighted_sum += data_point * responsibility_mat[data_point_idx][cluster_idx]
                self.means[cluster_idx] = weighted_sum / total_responsibility_per_cluster[cluster_idx]
            
            # Update covariance matrices
            for cluster_idx in range(self.k):
                weighted_sum = np.zeros((self.data.shape[1], self.data.shape[1]))
                for data_point_idx, data_point in enumerate(self.data):
                    diff = data_point - self.means[cluster_idx]
                    weighted_sum += np.outer(diff, diff) * responsibility_mat[data_point_idx][cluster_idx]
                self.cov_matrices[cluster_idx] = weighted_sum / total_responsibility_per_cluster[cluster_idx]
            
            # Calculate log likelihood
            log_likelihood = self.calc_log_likelihood()
            self.arr_log_likelihood.append(log_likelihood)

            epochs_taken += 1

            # Check for convergence
            if len(self.arr_log_likelihood) < 2:
                continue
            else:
                if abs(self.arr_log_likelihood[-1] - self.arr_log_likelihood[-2]) < 1e-3:
                    self.responsibility_mat = responsibility_mat
                    break
        return epochs_taken

    # Returns the list of weights, means and covariance matrices
    def getParams(self):
        return self.weights, self.means, self.cov_matrices

    # Returns the responsibility matrix
    def getMembership(self, i, c):
        return self.responsibility_mat

    # Returns the overall log likelihood of the fitted model
    def getLikelihood(self):
        overall_likelihood = 1
        for data_point in self.data:
            likelihood = 0
            for weight, mean, cov_matrix in zip(self.weights, self.means, self.cov_matrices):
                likelihood += weight * multivariate_normal.pdf(data_point, mean, cov_matrix, allow_singular=True)
            overall_likelihood *= likelihood
        return overall_likelihood
    
    # Returns log likelihood of the fitted model
    def calc_log_likelihood(self):
        return np.log(self.getLikelihood() + 1e-10)