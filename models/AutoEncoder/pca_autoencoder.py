import numpy as np
import torch
import torch.nn as nn

class PCA_Autoencoder(nn.Module):
    def __init__(self, n_components=None):
        super(PCA_Autoencoder, self).__init__()
        self.n_components = n_components
        self.eigenvectors = None
        self.mean = None

    def fit(self, data):
        # Flatten the 28x28 images to a 784-dimensional vectors
        data = data.reshape(data.shape[0], -1)  # Shape: (num_samples, 784)

        # Centering the data
        self.mean = np.mean(data, axis=0)
        centered_data = data - self.mean

        # Calculating the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Computing eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sorting eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Keep only the top n_components eigenvectors
        if self.n_components:
            eigenvectors = eigenvectors[:, :self.n_components]

        # Store the eigenvectors as a torch tensor
        self.eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)
        self.mean = torch.tensor(self.mean, dtype=torch.float32)

    def encode(self, data):
        # Flatten images and center data
        data = data.reshape(data.shape[0], -1)
        centered_data = torch.tensor(data, dtype=torch.float32) - self.mean
        # Project data onto the eigenvectors
        return torch.matmul(centered_data, self.eigenvectors).numpy()

    def forward(self, encoded_data):
        if self.eigenvectors is None:
            raise ValueError("PCA model is not fitted yet.")
        # Reconstruct data by reversing the projection
        reconstructed_data = torch.matmul(encoded_data, self.eigenvectors.T) + self.mean
        # Reshape back to the original 28x28 image shape
        return reconstructed_data.view(-1, 28, 28).numpy()