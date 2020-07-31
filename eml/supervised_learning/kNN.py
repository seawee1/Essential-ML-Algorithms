import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class KNearestNeighbors():
    def __init__(self, k=1, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_classes = np.max(y) + 1 # Assumes classes are indexed from 0 to n_classes - 1

    def predict_proba(self, X):
        # Compute pairwise L2 distance
        edm = euclidean_distances(X, self.X)
        # Take k smallest distances per input vector and determine class lables
        kNNs = self.y[np.argsort(edm, axis=1)[:, 0:self.k]]
        # Count class occurences per input vector, divide by k
        prob = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.n_classes), axis=1, arr=kNNs)
        prob = prob/self.k
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)
