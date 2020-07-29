import numpy as np
#from numpy.linalg import eig
from scipy import linalg

class DiscriminantAnalysis:

    def _prepare_data(self, X, y):
        # Estimate class priors
        unique, unique_counts = np.unique(y, return_counts=True)
        n_classes = len(unique)
        class_priors = unique_counts / len(y)

        # Split X into class subsets
        X_ = [X[y==i] for i in range(n_classes)]

        return X_, class_priors, n_classes

    def _compute_covariances(self, X_):
        # Estimate class centroids
        centroids = np.mean(X_, axis=1)

        # Class covariance matrices weighted by 1/(N_k - 1) (unbiased estimate)
        cov_ = [(1/(X_[i].shape[0]-1)) * (X_[i] - c).T.dot(X_[i] - c) for i, c in enumerate(centroids)]
        # Shared covariance matrix
        cov = (1/len(X_)) * np.sum(cov_, axis=0)

        return cov, cov_, centroids

    def _compute_scatter(self, X_):
        # Estimate class centroids and data centroid
        centroids = np.mean(X_, axis=1)
        data_centroid = np.mean(X_, axis=(0,1))

        # Between-class and within-class scatter matrices
        S_b = np.sum([X_[i].shape[0] *  np.outer(c - data_centroid, c-data_centroid) for i, c in enumerate(centroids)], axis=0)
        S_w = np.sum([(X_[i] - c).T.dot(X_[i] - c) for i, c in enumerate(centroids)], axis=0)

        return S_b, S_w


class LinearDiscriminantAnalysis(DiscriminantAnalysis):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        # Set target dimensionality if not given
        if self.n_components is None:
            self.n_components = min(self.n_classes - 1, X.shape[1]) if self.n_classes > 1 else 1

        X_, self.class_priors, self.n_classes = super()._prepare_data(X, y)
        cov, _, centroids = super()._compute_covariances(X_)

        if self.n_components < X.shape[1]:
            # Compute scatter matrices and solve Fisher eigenproblem
            S_b, S_w = super()._compute_scatter(X_)
            w, V = linalg.eig(S_b, S_w)
            w, V = w.real, V.real

            # Take self.n_components eigenvectors with largest eigenvalues as new basis
            self.V_sorted = V[(-w).argsort()]
            self.V_sorted = self.V_sorted[:self.n_components]
        else:
            self.V_sorted = np.identity(X.shape[1])

        # Compute mean and shared covariance of transformed class distributions
        cov = self.V_sorted.dot(cov).dot(self.V_sorted.T)
        centroids = centroids.dot(self.V_sorted.T)
        self.cov_inv = np.linalg.inv(cov)

    def transform(self, X):
        X_new = X.dot(self.V_sorted.T)
        return X_new

    def predict(self, X):
        X_new = self.transform(X)
        scores = X_new.dot(self.cov_inv).dot(self.centroids.T) - 0.5 * np.sum(self.centroids.dot(self.cov_inv) * self.centroids, axis=1).T + np.log(self.class_priors)
        pred = np.argmax(scores, axis=1)
        return pred

class QuadraticDiscriminantAnalysis(DiscriminantAnalysis):

    def fit(self, X, y):
        X_, self.class_priors, self.n_classes = super()._prepare_data(X, y)
        _, self.cov_, self.centroids = super()._compute_covariances(X_)

        # Inverse
        self.cov_inv_ = [np.linalg.inv(cov) for cov in self.cov_]

    # Apply decision function of class c
    def __decision_function(self, X, c):
        X_centered = X - self.centroids[c]
        scores = -0.5 * np.log(np.linalg.det(self.cov_[c])) - 0.5 * np.sum(X_centered.dot(self.cov_inv_[c]) * X_centered, axis=1).T + np.log(self.class_priors[c])
        return scores

    def predict(self, X):
        pred = np.array([self.__decision_function(X, i) for i in range(self.n_classes)])
        pred = np.argmax(pred, axis=0)
        return pred
