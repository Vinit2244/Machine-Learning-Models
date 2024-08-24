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
        self.true_vals = None   # True values of the variable to predict
        self.predicted_vals = None  # Predicted values of the variable to predict
    
    # Getter and Setter for variable to predict
    def set_predict_var(self, column_name):
        self.to_predict = column_name
    
    def get_predict_var(self):
        return self.to_predict
    
    # Set the features to use for making predictions
    def use_for_prediction(self, features):
        self.features = features

    # Returns the list of features used for making predictions
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

    # Load the training, testing and validation data
    def load_train_test_val_data(self, headers, train_data, test_data, validation_data):
        self.cols = headers
        self.train_data = train_data
        self.test_data = test_data
        self.validation_data = validation_data

    # Calculate the distance between a single data point and an array of data points
    def calc_distance(self, x1, x2_arr):
        if self.dist_metric == "manhattan":
            return np.sum(np.abs(x2_arr - x1), axis=1)
        elif self.dist_metric == "euclidean":
            return np.linalg.norm(x2_arr - x1, axis=1)
        elif self.dist_metric == "cosine":
            vector_norm = np.linalg.norm(x1)
            vectors_array_norm = np.linalg.norm(x2_arr, axis=1)
            dot_products = np.dot(x2_arr, x1)
            return dot_products / (vector_norm * vectors_array_norm)
        else:
            raise ValueError("Invalid distance metric")

    # Fit the model
    def fit(self):
        pass

    # Assumes the order and number of columns (except the label column) in test data is the same as in train data
    def predict(self, type_of_data):
        features_to_use_idx = [self.cols.index(feature) for feature in self.features]
        filtered_train_data = self.train_data[:, features_to_use_idx]
        filtered_to_predict_data = None
        train_data_labels = self.train_data[:, -1]

        if type_of_data == "test":
            # Refining testing data points
            filtered_to_predict_data = self.test_data[:, features_to_use_idx]
            self.true_vals = self.test_data[:, -1]
        elif type_of_data == "validation":
            # Refining validation data points
            filtered_to_predict_data = self.validation_data[:, features_to_use_idx]
            self.true_vals = self.validation_data[:, -1]

        # Converting them to float32 for optimisation
        filtered_train_data = np.float32(filtered_train_data)
        filtered_to_predict_data = np.float32(filtered_to_predict_data)

        predictions = []
        for data_point in filtered_to_predict_data:
            distances = self.calc_distance(data_point, filtered_train_data)

            distance_label_pairs = list(zip(distances, train_data_labels))
            sorted_pairs = sorted(distance_label_pairs, key=lambda x: x[0])
            top_k_labels = [label for _, label in sorted_pairs[:self.k]]
            
            '''
                The below line of code is given by ChatGPT
                Prompt: given a list of labels find the most common label in the list in 1 line of code
            '''
            # =============================================================================
            most_frequent_label = max(set(top_k_labels), key=top_k_labels.count)
            # =============================================================================
            predictions.append(most_frequent_label)
        
        self.predicted_vals = np.array(predictions)
        return predictions

    def get_metrics(self):
        if self.predicted_vals is None or self.true_vals is None:
            raise ValueError("Predictions have not been made yet - run the model atleast once before getting metrics")

        list_of_classes = list(set(self.true_vals))

        # Calculate the confusion matrix for each of the classes
        list_of_confusion_matrices = []

        for cls in list_of_classes:
            list_of_confusion_matrices.append(self.confusion_matrix(cls, self.true_vals, self.predicted_vals))

        # Pool the confusion matrices: Summing the matrices cell wise
        pooled_confusion_matrix = np.sum(list_of_confusion_matrices, axis=0)
        
        # Calculate the accuracy of the model
        accuracy = self.accuracy(self.true_vals, self.predicted_vals)

        list_of_individual_precisions = self.precision(list_of_confusion_matrices)
        list_of_individual_recalls = self.recall(list_of_confusion_matrices)
        list_of_individual_f1_scores = self.f1_score(list_of_individual_precisions, list_of_individual_recalls)

        macro_precision = np.mean(np.array(list_of_individual_precisions))
        macro_recall = np.mean(np.array(list_of_individual_recalls))
        macro_f1_score = np.mean(np.array(list_of_individual_f1_scores))

        micro_precision = self.precision([pooled_confusion_matrix])[0]
        micro_recall = self.recall([pooled_confusion_matrix])[0]
        micro_f1_score = self.f1_score([micro_precision], [micro_recall])[0]

        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1_score": macro_f1_score,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1_score": micro_f1_score
        }
