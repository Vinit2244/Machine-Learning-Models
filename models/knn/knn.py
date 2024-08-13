class KNN:
    def __init__(self, k, distance_metric_function):
        self.k = k
        self.distance_metric = distance_metric_function
    
    # Getter and Setter for k and distance_metric
    def get_k(self):
        return self.k

    def set_k(self, k):
        self.k = k

    def get_distance_metric(self):
        return self.distance_metric
    
    def set_distance_metric(self, distance_metric_function):
        self.distance_metric = distance_metric_function

    # Note that here distance_metric is function that takes two arguments and returns a float
    def calc_distance(self, a, b):
        return self.distance_metric(a, b)