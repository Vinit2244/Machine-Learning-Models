import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class KDE:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data = None
    
    def fit(self, data):
        """Store the data points."""
        self.data = np.array(data)
    
    def _kernel_function(self, u):
        """Select the kernel function based on the hyperparameter."""
        if self.kernel == 'box':
            return 0.5 * (np.abs(u) <= 1)
        elif self.kernel == 'gaussian':
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        elif self.kernel == 'triangular':
            return (1 - np.abs(u)) * (np.abs(u) <= 1)
        else:
            raise ValueError("Unsupported kernel type")
    
    def predict(self, x):
        """Estimate the density at a given point x."""
        x = np.array(x)
        n, d = self.data.shape
        densities = 0
        
        for xi in self.data:
            u = (x - xi) / self.bandwidth
            densities += np.prod(self._kernel_function(u))
        
        return densities / (n * self.bandwidth**d)
    
    def visualize(self, plot_type='3d', save_as=None):
        """Visualize the density for 2D data as a 3D plot or 2D contour plots."""
        if self.data.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        # Determine the plotting grid range
        x_min, y_min = np.min(self.data, axis=0) - 1
        x_max, y_max = np.max(self.data, axis=0) + 1
        
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
        Z = np.zeros_like(X)
        
        # Calculate the density (Z) for the grid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.predict([X[i, j], Y[i, j]])
        
        if plot_type == '3d':
            # 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Density')
            plt.title("3D Density Plot")
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
                plt.close(fig)

        elif plot_type == 'contour':
            # Create a figure with two subplots for side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Standard contour plot
            axes[0].set_title('Standard 2D Contour Plot')
            axes[0].contour(X, Y, Z, levels=10, cmap='viridis')
            axes[0].scatter(self.data[:, 0], self.data[:, 1], c='black', s=5, alpha=0.5)
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')

            # Filled contour plot
            axes[1].set_title('Filled 2D Contour Plot')
            c = axes[1].contourf(X, Y, Z, levels=50, cmap='viridis')
            fig.colorbar(c, ax=axes[1], label='Density')
            axes[1].scatter(self.data[:, 0], self.data[:, 1], c='black', s=5, alpha=0.5)
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')

            # Adjust layout
            plt.tight_layout()
            if save_as is None:
                plt.show()
            else:
                plt.savefig(save_as)
                plt.close()