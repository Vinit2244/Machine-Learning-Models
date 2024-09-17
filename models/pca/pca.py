import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.data = None
        self.principle_components = None
        self.mean_of_data = None
        self.transformed_data = None

    def load_data(self, data):
        self.data = data

    def get_eigenvalues_eigenvectors(self):
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
        sorted_eigenvectors = np.complex64(eigenvectors.T[sorted_indices_desc]).real
        sorted_eigenvalues = np.complex64(eigenvalues[sorted_indices_desc]).real
        
        return sorted_eigenvalues, sorted_eigenvectors

    def fit(self):
        sorted_eigenvalues, sorted_eigenvectors = self.get_eigenvalues_eigenvectors()
        
        # Select the top n_components eigenvectors
        self.principle_components = np.complex128(sorted_eigenvectors[:self.n_components]).real # First converting complex to real

    def transform(self):
        centered_data = self.data - self.mean_of_data
        self.transformed_data = np.matmul(centered_data, self.principle_components.T)
        return self.transformed_data
    
    def checkPCA(self, transformed_data):
        reconstructed_data = np.matmul(transformed_data, self.principle_components) + self.mean_of_data
        reconstruction_error = np.mean((self.data - reconstructed_data) ** 2)
        if reconstruction_error < 0.1:
            return True
        else:
            return False