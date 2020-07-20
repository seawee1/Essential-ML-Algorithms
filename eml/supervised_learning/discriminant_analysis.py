import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self, n_iter=100, lr=0.001, solver='mle'):
        self.n_iter = n_iter
        self.lr = lr

    def fit(self, X, y):
        # Estimate class priors
        unique, unique_counts = np.unique(y, return_counts=True)
        self.n_classes = len(unique)
        self.class_priors = unique_counts / len(y)

        # Split X into subsets of same class
        X_c = [X[y==i] for i in range(self.n_classes)]

        # Estimate class means
        mu_c = [np.mean(x, axis=0) for x in X_c]
        print(mu_c[0].shape)

        # Cov
        cov = np.sum()
