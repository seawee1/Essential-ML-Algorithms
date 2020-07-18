import numpy as np
from numpy.random import multivariate_normal

def binary_classification(n_samples=1000):
    # Sample datapoints
    X_0 = multivariate_normal([-1, 1], [[1, 0],[0,1]], int(n_samples/2))
    X_1 = multivariate_normal([1, -1], [[1, 0],[0,1]], int(n_samples/2))
    X = np.vstack((X_0, X_1))

    # Create labels
    y = np.zeros((1000,))
    y[int(n_samples/2):] = 1

    # Shuffle
    idx = np.arange(1000)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X, y

def linear_regression(n_samples=100):
    data = multivariate_normal([0, 0], [[2, 0.8], [0.8, 0.5]], n_samples)
    X = data[:,0]
    y = data[:,1]

    return X, y
