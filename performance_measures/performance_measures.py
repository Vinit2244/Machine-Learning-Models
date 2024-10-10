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
    
    def accuracy(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                n_correct += 1
        total = len(y_true)
        return n_correct / total
    
    def get_precision_recall_f1_onehot(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_classes = y_true.shape[1]
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        list_of_confusion_matrices = []
        for cls in range(n_classes):
            confusion_matrix = self.confusion_matrix(cls, y_true, y_pred)
            list_of_confusion_matrices.append(confusion_matrix)
        list_of_precisions = self.precision(list_of_confusion_matrices)
        list_of_recalls = self.recall(list_of_confusion_matrices)
        list_of_f1_scores = self.f1_score(list_of_precisions, list_of_recalls)
        return list_of_precisions, list_of_recalls, list_of_f1_scores

    # Does not return matrix values in percentages - as usual confusion matrix has, but returns the counts
    def confusion_matrix(self, cls, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        cls_confusion_matrix = np.zeros((2, 2))

        for i in range(len(y_true)):
            if y_true[i] == cls:
                if y_pred[i] == cls:
                    cls_confusion_matrix[0][0] += 1
                else:
                    cls_confusion_matrix[0][1] += 1
            else:
                if y_pred[i] == cls:
                    cls_confusion_matrix[1][0] += 1
                else:
                    cls_confusion_matrix[1][1] += 1
        return cls_confusion_matrix
    
    def precision(self, list_of_confusion_matrices):
        list_of_micro_precisions = []
        for confusion_matrix in list_of_confusion_matrices:
            tp = confusion_matrix[0][0]
            fp = confusion_matrix[0][1]
            micro_precision = tp / (tp + fp + 1e-10) # 1e-10 is added to avoid division by zero
            list_of_micro_precisions.append(micro_precision)
        return list_of_micro_precisions
    
    def recall(self, list_of_confusion_matrices):
        list_of_micro_recalls = []
        for confusion_matrix in list_of_confusion_matrices:
            tp = confusion_matrix[0][0]
            fn = confusion_matrix[1][0]
            micro_recall = tp / (tp + fn + 1e-10) # 1e-10 is added to avoid division by zero
            list_of_micro_recalls.append(micro_recall)
        return list_of_micro_recalls
    
    def f1_score(self, list_of_precisions, list_of_recall):
        list_of_f1_scores = []
        for i in range(len(list_of_precisions)):
            precision = list_of_precisions[i]
            recall = list_of_recall[i]
            f1 = (2 * precision * recall) / (precision + recall + 1e-10) # 1e-10 is added to avoid division by zero
            list_of_f1_scores.append(f1)
        return list_of_f1_scores
    
    def cross_entropy(self, y_true, y_hat):
        y_true = np.array(y_true)
        y_hat = np.array(y_hat)
        cross_entropy = -np.mean(y_true * np.log(y_hat + 1e-10))
        return cross_entropy
    
    def RMSE(self, y_true, y_hat):
        mse = self.MSE(y_true, y_hat)
        return np.sqrt(mse)
    
    def MAE(self, y_true, y_hat):
        y_true = np.array(y_true)
        y_hat = np.array(y_hat)
        mae = np.mean(np.abs(y_true - y_hat))
        return mae
    
    def R2(self, y_true, y_hat):
        y_true = np.array(y_true)
        y_hat = np.array(y_hat)
        mean = np.mean(y_true)
        ss_total = np.sum((y_true - mean) ** 2)
        ss_res = np.sum((y_true - y_hat) ** 2)
        r2 = 1 - (ss_res / (ss_total + 1e-10))
        return r2