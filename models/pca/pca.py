import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.data = None
        self.principle_components = None
        self.mean_of_data = None
        self.transformed_data = None

    def load_data(self, data):
        self.data = data
        # self.principle_components = np.full((self.n_components, self.data.shape[1]), np.nan)

    def fit(self):
        # Centering the data
        self.mean_of_data = np.mean(self.data, axis=0)
        X_centered = self.data - self.mean_of_data
        
        # Covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)
        sorted_indices_desc = sorted_indices[::-1]
        sorted_eigenvectors = eigenvectors.T[sorted_indices_desc]
        
        # Select the top n_components eigenvectors
        self.principle_components = np.complex128(sorted_eigenvectors[:self.n_components]).real # First converting complex to real

    def transform(self):
        centered_data = self.data - self.mean_of_data
        self.transformed_data = np.matmul(centered_data, self.principle_components.T)
        return self.transformed_data
    
    def checkPCA(self):
        return self.transformed_data.shape[1] == self.n_components