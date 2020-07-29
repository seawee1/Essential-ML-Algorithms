import numpy as np
from numpy.random import multivariate_normal, rand

def gaussians(n_classes, samples_per_class, mean_covs):
    X = np.vstack([multivariate_normal(mean_covs[i][0], mean_covs[i][1], samples_per_class) for i in range(n_classes)])
    y = np.array([[i] * samples_per_class for i in range(n_classes)]).flatten()

    # Shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    return X, y

def binary_gaussian(samples_per_class=500):
    mean_covs = [
        [[-1, 1], [[1, 1],[1,2]]],
        [[1, -1], [[1, 1],[1,2]]]
    ]

    return gaussians(2, samples_per_class, mean_covs)

def multi_gaussian_3d(samples_per_class=500):
    mean_covs = [
        [[-5, -5, 5], [[2, 0, 0],[0,1,0], [0,0,1]]],
        [[5, -5, 5], [[2, 0, 0],[0,1,0], [0,0,1]]],
        [[0, 5, 5], [[2, 0, 0],[0,1,0], [0,0,1]]]
    ]

    return gaussians(3, samples_per_class, mean_covs)

def multi_gaussian(samples_per_class=500):
    mean_covs = [
        [[4, 5], [[2, -0.6],[-0.6, 1]]],
        [[-4, 7], [[1.5, 0],[0, 2]]],
        [[-4, -2], [[2, 0.9],[0.9, 0.5]]],
        [[4, 0], [[1, -0.1],[-0.1, 0.5]]],
    ]

    return gaussians(4, samples_per_class, mean_covs)

def linear_regression(n_samples=100):
    data = multivariate_normal([0, 0], [[2, 0.8], [0.8, 0.5]], n_samples)
    X = data[:,0]
    y = data[:,1]

    return X, y
