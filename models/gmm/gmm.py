import numpy as np
import random
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, k, seed=None, epsilon=1e-10):
        self.k = k                      # Number of gaussian models/clusters
        self.data = None                # 2D numpy array
        self.weights = None             # Weights of each cluster
        self.means = None               # Means of each cluster
        self.cov_matrices = None        # Covariance matrices of each cluster (full covariance matrix)
        self.responsibility_mat = None  # Responsibility matrix
        self.arr_log_likelihood = []        # Likelihood of the model
        self.seed = seed                # Seed for reproducibility
        self.epsilon = epsilon

    def load_data(self, data):
        self.data = data

    def initialise_params(self):
        self.responsibility_mat = np.zeros((self.data.shape[0], self.k))

        # Initialise weights
        if self.seed is not None:
            np.random.seed(self.seed)
        
        weights = np.random.rand(self.k)         # Initialise some random values
        self.weights = weights / np.sum(weights) # Normalise the weights

        # Initialise means
        indexes = random.sample(range(0, self.data.shape[0]), self.k) # Randomly select k indexes
        self.means = self.data[indexes]                               # Assign the data points at those indexes as means

        # Initialise variances
        self.cov_matrices = [np.eye(self.data.shape[1]) for _ in range(self.k)]

    # EM algorithm
    def fit(self):
        self.initialise_params()

        epochs_taken = 0

        while True:
            # Expectation Step
            # for data_point_idx, data_point in enumerate(self.data):
            #     for cluster_idx in range(self.k):
            #         numerator = self.weights[cluster_idx] * multivariate_normal.pdf(data_point, self.means[cluster_idx], self.cov_matrices[cluster_idx], allow_singular=True)
            #         denominator = 0
            #         for i in range(self.k):
            #             denominator += self.weights[i] * multivariate_normal.pdf(data_point, self.means[i], self.cov_matrices[i], allow_singular=True)
            #         if denominator != 0:
            #             self.responsibility_mat[data_point_idx][cluster_idx] = numerator / denominator
            #         else:
            #             self.responsibility_mat[data_point_idx][cluster_idx] = 0
            # Compute the PDF values for all data points across all clusters in a vectorized way
            pdf_values = np.zeros((self.data.shape[0], self.k))

            for cluster_idx in range(self.k):
                pdf_values[:, cluster_idx] = multivariate_normal.pdf(self.data, self.means[cluster_idx], self.cov_matrices[cluster_idx], allow_singular=True)

            # Calculate the numerator (weights * pdf for each cluster)
            numerator = self.weights * pdf_values

            # Calculate the denominator (sum of weighted pdfs across clusters for each data point)
            denominator = np.sum(numerator, axis=1, keepdims=True)

            # Update the responsibility matrix (r_ic) for each data point and cluster
            self.responsibility_mat = np.divide(numerator, denominator, where=(denominator != 0), out=np.zeros_like(numerator))

            
            # Maximisation Step
            # for j in range(self.k):
            #     Nk = np.sum(self.responsibility_mat[:, j])
            #     self.means[j] = np.sum(self.responsibility_mat[:, j].reshape(-1, 1) * self.data, axis=0) / Nk
            #     self.cov_matrices[j] = np.dot((self.responsibility_mat[:, j].reshape(-1, 1) * (self.data - self.means[j])).T, (self.data - self.means[j])) / Nk
            #     self.weights[j] = Nk / self.data.shape[0]

            # list_of_N = list()
            # total_number_of_points = self.data.shape[0]

            # for cluster_idx in range(self.k):
            #     N_k = 0
            #     for data_point_idx, data_point in enumerate(self.data):
            #         N_k += self.responsibility_mat[data_point_idx][cluster_idx]
            #     list_of_N.append(N_k)

            # # Updating Means
            # for cluster_idx in range(self.k):
            #     weighted_sum_of_data_points = 0
            #     for data_point_idx, data_point in enumerate(self.data):
            #         weighted_sum_of_data_points += self.responsibility_mat[data_point_idx][cluster_idx] * data_point
            #     self.means[cluster_idx] = weighted_sum_of_data_points / list_of_N[cluster_idx]
            
            # # Updating covariance matrices
            # for cluster_idx in range(self.k):
            #     weighted_sum_of_cov_matrices = np.zeros((self.data.shape[1], self.data.shape[1]))
            #     for data_point_idx, data_point in enumerate(self.data):
            #         weighted_sum_of_cov_matrices += self.responsibility_mat[data_point_idx][cluster_idx] * (np.outer(data_point - self.means[cluster_idx], data_point - self.means[cluster_idx]))
            #     self.cov_matrices[cluster_idx] = weighted_sum_of_cov_matrices / list_of_N[cluster_idx]

            # Compute N_k for all clusters at once (the sum of responsibilities for each cluster)
            list_of_N = np.sum(self.responsibility_mat, axis=0)

            total_number_of_points  = self.data.shape[0]

            # Updating means (weighted sum of data points for each cluster)
            self.means = (self.responsibility_mat.T @ self.data) / list_of_N[:, np.newaxis]

            # Updating covariance matrices
            for cluster_idx in range(self.k):
                diff = self.data - self.means[cluster_idx]  # Shape (n_points, n_features)
                weighted_diff = self.responsibility_mat[:, cluster_idx][:, np.newaxis] * diff  # Shape (n_points, n_features)
                self.cov_matrices[cluster_idx] = (weighted_diff.T @ diff) / list_of_N[cluster_idx]


            # Updating Weights
            self.weights = np.array(list_of_N) / total_number_of_points

            log_likelihood = self.get_log_likelihood()
            self.arr_log_likelihood.append(log_likelihood)

            epochs_taken += 1

            # Check for convergence
            if len(self.arr_log_likelihood) < 2:
                continue
            else:
                if abs(self.arr_log_likelihood[-1] - self.arr_log_likelihood[-2]) < 1:
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
        log_likelihood = self.get_log_likelihood()
        return np.exp(log_likelihood)
    
    def visualise(self, x_range=(-3, 3), y_range=(-3, 3), grid_density=100, save_as=None, plot_type='contour'):
        if self.data.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        # Create grid for contour plotting
        x = np.linspace(x_range[0], x_range[1], grid_density)
        y = np.linspace(y_range[0], y_range[1], grid_density)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        # Initialize a combined density for plotting multiple Gaussians
        Z_total = np.zeros(X.shape)

        for mean, cov in zip(self.means, self.cov_matrices):
            rv = multivariate_normal(mean, cov)
            Z_total += rv.pdf(pos)

        if plot_type == 'contour':
            # Create a figure with side-by-side subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Type 1: Standard contour plot
            axes[0].set_title('2D Contour Plot of GMM (Type 1)')
            for mean, cov in zip(self.means, self.cov_matrices):
                rv = multivariate_normal(mean, cov)
                axes[0].contour(X, Y, rv.pdf(pos), levels=10, cmap='viridis')
            if self.data is not None:
                axes[0].scatter(self.data[:, 0], self.data[:, 1], c='red', s=10, label='Data Points')
            axes[0].set_xlabel('X-axis')
            axes[0].set_ylabel('Y-axis')
            axes[0].legend()
            axes[0].grid(True)

            # Type 2: Filled contour plot
            axes[1].set_title('2D Filled Contour Plot of GMM (Type 2)')
            c = axes[1].contourf(X, Y, Z_total, levels=50, cmap='viridis')
            fig.colorbar(c, ax=axes[1], label='Density')
            if self.data is not None:
                axes[1].scatter(self.data[:, 0], self.data[:, 1], c='black', s=5, alpha=0.5)
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')

            # Adjust layout and show or save the plot
            plt.tight_layout()
            if save_as is None:
                plt.show()
            else:
                plt.savefig(save_as)
                plt.close()

        elif plot_type == '3d':
            # 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z_total, cmap='viridis', edgecolor='none')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Density')
            plt.title("3D Density Plot of GMM")

            if save_as is None:
                plt.show()
            else:
                # Create a rotating animation and save as GIF
                def update(frame):
                    ax.view_init(elev=30, azim=frame)
                    return ax,

                frames = np.arange(0, 360, 2)
                ani = FuncAnimation(fig, update, frames=frames, interval=50)
                ani.save(save_as, writer='pillow')
                plt.close()

    # Returns log likelihood of the fitted model
    def get_log_likelihood(self):
        overall_log_likelihood = 0
        for data_point in self.data:
            log_likelihood_per_data_point = 0
            for cluster_idx in range(self.k):
                log_likelihood_per_data_point += self.weights[cluster_idx] * multivariate_normal.pdf(data_point, self.means[cluster_idx], self.cov_matrices[cluster_idx], allow_singular=True)
            overall_log_likelihood += np.log(log_likelihood_per_data_point + self.epsilon)
        return overall_log_likelihood