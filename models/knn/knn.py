import numpy as np
from performance_measures.performance_measures import PerformanceMetrics

class KNN(PerformanceMetrics):
    def __init__(self, k, dist_metric):
        self.k = k
        self.dist_metric = dist_metric # manhattan, euclidean, cosine
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.to_predict = None  # Column name to predict
        self.features = None    # Features to use for prediction
        self.cols = None        # Column names
    
    def set_predict_var(self, column_name):
        self.to_predict = column_name
    
    def get_predict_var(self):
        return self.to_predict
    
    def use_for_prediction(self, features):
        self.features = features

    def get_features_used_for_prediction(self):
        return self.features

    # Getter and Setter for k
    def get_k(self):
        return self.k
    
    def set_k(self, new_k):
        self.k = new_k

    # Getter and setter for distance metric
    def get_dist_metric(self):
        return self.dist_metric
    
    def set_dist_metric(self, new_dist_metric):
        self.dist_metric = new_dist_metric

    def load_train_data(self, headers, data):
        self.cols = headers
        self.train_data = data

    # x1 and x2 are both numpy arrays of equal length
    def calc_distance(self, x1, x2):
        if self.dist_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.dist_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.dist_metric == "cosine":
            return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        else:
            raise ValueError("Invalid distance metric")
        
    def predict_single(self, test_point, train_data, labels_train_data):
        distances = []
        for idx, train_point in enumerate(train_data):
            distances.append((self.calc_distance(test_point, train_point), idx))
        
        top_k_labels_idx = np.int32(np.array(sorted(distances)[:self.k]))[:, -1].flatten().tolist()
        top_k_labels = [labels_train_data[i] for i in top_k_labels_idx]
        '''
            The below line of code is given by ChatGPT
            Prompt: given a list of labels find the most common label in the list in 1 line of code
        '''
        # =============================================================================
        return max(set(top_k_labels), key=top_k_labels.count)
        # =============================================================================

    
    # Assumes the order and number of columns (except the label column) in test data is the same as in train data
    def predict(self, test_data):
        # Refining training and testing data points
        features_to_use_idx = [self.cols.index(feature) for feature in self.features]
        filtered_train_data = self.train_data[:, features_to_use_idx]
        labels_train_data = self.train_data[:, -1]
        filtered_test_data = test_data[:, features_to_use_idx]

        batch_size = 500

        print(filtered_test_data.shape, filtered_train_data.shape)

        n_test = filtered_test_data.shape[0]
        n_train = filtered_train_data.shape[0]
        distance_matrix = np.zeros((n_test, n_train))

        # Iterate over test points in batches
        for i in range(0, n_test, batch_size):
            # Select the current batch of test points
            test_batch = filtered_test_data[i:i + batch_size]

            # Compute the distance for the current test batch against all train points
            # Broadcasting to create a 3D array of differences
            diff = test_batch[:, np.newaxis, :] - filtered_train_data[np.newaxis, :, :]
            
            # Now diff has shape (batch_size, n_train, n_features)
            
            # Apply the custom distance function in a vectorized way
            for k in range(diff.shape[0]):  # Iterating over the test batch
                distance_matrix[i + k, :] = np.apply_along_axis(
                    lambda x: self.calc_distance(x, np.zeros_like(x)),
                    1,
                    diff[k]
                )

        print(distance_matrix.shape)
        # # If you need all differences in a single array
        # all_differences = np.concatenate(all_differences, axis=1)

        # predictions = []
        # for data_point in filtered_test_data:
        #     predictions.append(self.predict_single(data_point, filtered_train_data, labels_train_data.tolist()))

        # return predictions
