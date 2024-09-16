import numpy as np
import random
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, seed=None):
        self.k = k                      # Number of gaussian models/clusters
        self.data = None                # 2D numpy array
        self.weights = None             # Weights of each cluster
        self.means = None               # Means of each cluster
        self.cov_matrices = None        # Covariance matrices of each cluster (full covariance matrix)
        self.responsibility_mat = None  # Responsibility matrix
        self.arr_log_likelihood = None  # Log likelihood of the model
        self.seed = seed                # Seed for reproducibility
        self.epsilon = 1e-100           # Small value to avoid division by zero

    def load_data(self, data):
        self.data = data

    def initialise_params(self):
        # Initialise weights
        weights = np.random.rand(self.k)         # Initialise some random values
        self.weights = weights / np.sum(weights) # Normalise the weights

        # Initialise means
        indexes = random.sample(range(0, self.data.shape[0]), self.k) # Randomly select k indexes
        self.means = self.data[indexes]                               # Assign the data points at those indexes as means

        # Initialise variances
        # if no of samples < dimensions, then use diagonal matrix (each feature independent of other features)
        if self.data.shape[0] < self.data.shape[1]:
            self.cov_matrices = np.array([np.diag(np.random.rand(self.data.shape[1])) for _ in range(self.k)])
            self.cov_matrices = self.cov_matrices + np.eye(self.cov_matrices[0].shape[0]) * self.epsilon  # Regularization to avoid singularity
        # if no of samples > dimensions, then use full covariance matrix
        else:
            self.cov_matrices = []
            for _ in range(self.k):
                A = np.random.rand(self.data.shape[1], self.data.shape[1])
                cov_matrix = np.dot(A, A.T)
                # cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * self.epsilon  # Regularization to avoid singularity
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
                responsibility_mat[data_point_idx] /= (np.sum(responsibility_mat[data_point_idx]) + self.epsilon)

            # Maximization step
            # Total responsibility assigned to each cluster
            total_responsibility_per_cluster = np.sum(responsibility_mat, axis=0) # Summation over i of r_ik
            total_responsibility = np.sum(total_responsibility_per_cluster)
            total_responsibility_per_cluster /= total_responsibility + self.epsilon

            # Update weights
            self.weights = total_responsibility_per_cluster / self.data.shape[0]

            # Update means
            for cluster_idx in range(self.k):
                # Weighted sum of all data points assigned to a cluster
                weighted_sum = np.zeros(self.data.shape[1])
                for data_point_idx, data_point in enumerate(self.data):
                    weighted_sum += data_point * responsibility_mat[data_point_idx][cluster_idx]
                self.means[cluster_idx] = weighted_sum / (total_responsibility_per_cluster[cluster_idx] + self.epsilon)

            # Update covariance matrices
            for cluster_idx in range(self.k):
                weighted_sum = np.zeros((self.data.shape[1], self.data.shape[1]))
                for data_point_idx, data_point in enumerate(self.data):
                    diff = data_point - self.means[cluster_idx]
                    weighted_sum += np.outer(diff, diff) * responsibility_mat[data_point_idx][cluster_idx]
                self.cov_matrices[cluster_idx] = weighted_sum / (total_responsibility_per_cluster[cluster_idx] + self.epsilon)

            # Calculate log likelihood
            log_likelihood = self.get_log_likelihood()
            self.arr_log_likelihood.append(log_likelihood)

            epochs_taken += 1

            # Check for convergence
            if len(self.arr_log_likelihood) < 2:
                continue
            else:
                if abs(self.arr_log_likelihood[-1] - self.arr_log_likelihood[-2]) < 1e-3:
                    self.responsibility_mat = responsibility_mat
                    break
                else:
                    continue
        return epochs_taken

    # Returns the list of weights, means and covariance matrices
    def getParams(self):
        return self.weights, self.means, self.cov_matrices

    # Returns the responsibility matrix
    def getMembership(self, i=None, c=None):
        if i is not None and c is not None:
            return self.responsibility_mat[i][c]
        else:
            return self.responsibility_mat

    # Returns the overall log likelihood of the fitted model
    def getLikelihood(self):
        likelihood_for_complete_dataset = 1
        for data_point in self.data:
            likelihood_per_data_point = 0

            # Sum of weighted probabilities of a data point belonging to each cluster
            for weight, mean, cov_matrix in zip(self.weights, self.means, self.cov_matrices):
                likelihood_per_data_point += weight * multivariate_normal.pdf(data_point, mean, cov_matrix, allow_singular=True)

            # Multiply the likelihood of each data point to get the likelihood of the complete dataset
            likelihood_for_complete_dataset *= likelihood_per_data_point
        return likelihood_for_complete_dataset
    
    # Returns log likelihood of the fitted model
    def get_log_likelihood(self):
        return np.log(self.getLikelihood() + self.epsilon)




# import numpy as np
# from scipy.stats import multivariate_normal

# class GMM:
#     def __init__(self, k, seed=None, max_iter=100, tol=1e-3):
#         self.k = k  # Number of clusters (Gaussian components)
#         self.max_iter = max_iter  # Maximum number of iterations
#         self.tol = tol  # Tolerance for convergence
        
#         self.data = None
#         self.weights = None
#         self.means = None
#         self.cov_matrices = None
#         self.responsibility_mat = None
#         self.log_likelihoods = []
#         self.seed = seed

#     def load_data(self, data):
#         self.data = data

#     def initialise_params(self):
#         n_samples, n_features = self.data.shape

#         # Initialize weights uniformly
#         self.weights = np.ones(self.k) / self.k

#         # Initialize means by sampling k random points from the data
#         if self.seed is not None:
#             np.random.seed(self.seed)
#         random_indices = np.random.choice(n_samples, self.k, replace=False)
#         self.means = self.data[random_indices]

#         # Initialize covariance matrices as identity matrices
#         self.cov_matrices = np.array([np.eye(n_features) for _ in range(self.k)])

#     def expectation_step(self):
#         """E-step: Compute the responsibility matrix"""
#         n_samples, n_features = self.data.shape
#         self.responsibility_mat = np.zeros((n_samples, self.k))

#         for cluster_idx in range(self.k):
#             mean = self.means[cluster_idx]
#             cov_matrix = self.cov_matrices[cluster_idx]
#             weight = self.weights[cluster_idx]
#             pdf_vals = multivariate_normal.pdf(self.data, mean=mean, cov=cov_matrix, allow_singular=True)
#             self.responsibility_mat[:, cluster_idx] = weight * pdf_vals

#         # Normalize responsibilities for each data point
#         self.responsibility_mat /= (self.responsibility_mat.sum(axis=1, keepdims=True) + 1e-10)

#     def maximization_step(self):
#         """M-step: Update the weights, means, and covariance matrices"""
#         n_samples, n_features = self.data.shape
#         total_responsibility_per_cluster = self.responsibility_mat.sum(axis=0)

#         # Update weights
#         self.weights = total_responsibility_per_cluster / n_samples

#         # Update means
#         self.means = np.dot(self.responsibility_mat.T, self.data) / (total_responsibility_per_cluster[:, np.newaxis] + 1e-10)

#         # Update covariance matrices
#         for cluster_idx in range(self.k):
#             diff = self.data - self.means[cluster_idx]
#             weighted_diff = diff.T @ (diff * self.responsibility_mat[:, cluster_idx][:, np.newaxis])
#             self.cov_matrices[cluster_idx] = weighted_diff / (total_responsibility_per_cluster[cluster_idx] + 1e-10)
#             # Add small value to diagonal for numerical stability (regularization)
#             self.cov_matrices[cluster_idx] += np.eye(n_features) * 1e-6

#     def compute_log_likelihood(self):
#         """Compute the log likelihood of the current model"""
#         n_samples = self.data.shape[0]
#         log_likelihood = 0
#         for i in range(n_samples):
#             likelihood_per_point = 0
#             for cluster_idx in range(self.k):
#                 mean = self.means[cluster_idx]
#                 cov_matrix = self.cov_matrices[cluster_idx]
#                 weight = self.weights[cluster_idx]
#                 likelihood_per_point += weight * multivariate_normal.pdf(self.data[i], mean=mean, cov=cov_matrix, allow_singular=True)
#             log_likelihood += np.log(likelihood_per_point + 1e-10)
#         return log_likelihood

#     def fit(self):
#         """Fit the GMM model to the data using the EM algorithm"""
#         self.initialise_params()

#         for i in range(self.max_iter):
#             prev_log_likelihood = self.compute_log_likelihood()

#             # E-step
#             self.expectation_step()

#             # M-step
#             self.maximization_step()

#             # Calculate log likelihood
#             log_likelihood = self.compute_log_likelihood()
#             self.log_likelihoods.append(log_likelihood)

#             # Check for convergence
#             if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
#                 print(f"Converged after {i + 1} iterations")
#                 break
#         else:
#             print(f"Reached maximum iterations ({self.max_iter})")

#     def get_params(self):
#         """Return the weights, means, and covariance matrices"""
#         return self.weights, self.means, self.cov_matrices

#     def predict_proba(self):
#         """Return the responsibility matrix (soft cluster assignments)"""
#         return self.responsibility_mat

#     def predict(self):
#         """Return the most likely cluster for each data point (hard assignment)"""
#         return np.argmax(self.responsibility_mat, axis=1)

#     def get_log_likelihood(self):
#         """Return the log likelihood of the final model"""
#         return self.log_likelihoods[-1] if self.log_likelihoods else None
