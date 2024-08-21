import numpy as np

class PerformanceMetrics:
    def MSE(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_error = y_true - y_pred
        mse = np.mean(y_error ** 2)
        return mse
    
    def variance(self, arr):
        arr = np.array(arr)
        mean = np.mean(arr)
        var = np.mean((arr - mean) ** 2)
        return var
    
    def standard_deviation(self, arr):
        var = self.variance(arr)
        return np.sqrt(var)