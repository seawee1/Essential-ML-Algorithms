import numpy as np

class DiscriminantAnalysis:
    def __init__(self, n_iter=100, lr=0.001):
        self.n_iter = n_iter
        self.lr = lr

    def _prepare_data(self, X, y):
        # Estimate class priors
        unique, unique_counts = np.unique(y, return_counts=True)
        n_classes = len(unique)
        class_priors = unique_counts / len(y)

        # Split X into class subsets
        X_ = [X[y==i] for i in range(n_classes)]

        return X_, class_priors

    def _compute_statistics(self, X_):
        # Estimate class means
        centroids = np.array([np.mean(x, axis=0) for x in X_])

        # Center data w.r.t class centroids
        X_ = [X_[i] - centroids[i] for i in range(centroids.shape[0])]

        # Class covariance matrices weighted by 1/(N - K)
        cov_ = [X_[i].T.dot(X_[i]) for i in range(centroids.shape[0])]

        return centroids, cov_


class LinearDiscriminantAnalysis(DiscriminantAnalysis):
    def __init__(self, n_iter=100, lr=0.001, solver='standard'):
        super().__init__(n_iter=n_iter, lr=lr)
        self.solver=solver

    def fit(self, X, y):
        X_, self.class_priors = super()._prepare_data(X, y)
        self.centroids, cov_ = super()._compute_statistics(X_)

        # Weight class covariance matrices by 1/(N - K)
        cov_ = [(1/(X.shape[0] - self.centroids.shape[0])) * cov_[i] for i in range(len(cov_))]

        # Shared covariance matrix
        cov = np.sum(cov_, axis=0)

        # Inverse
        self.cov_inv = np.linalg.inv(cov)

    def predict(self, X):
        # Decision functions
        scores = X.dot(self.cov_inv).dot(self.centroids.T) - (1/2) * np.sum(self.centroids.dot(self.cov_inv) * self.centroids, axis=1).T + np.log(self.class_priors)
        pred = np.argmax(scores, axis=1)
        return pred
