import numpy as np
#from numpy.linalg import eig
from scipy import linalg
from scipy.stats import multivariate_normal

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
        centroids = np.array([np.mean(x, axis=0) for x in X_])

        # Class covariance matrices weighted by 1/(N_k - 1) (unbiased estimate)
        cov_ = [(1/(X_[i].shape[0]-1)) * (X_[i] - c).T.dot(X_[i] - c) for i, c in enumerate(centroids)]
        # Shared covariance matrix
        cov = (1/len(X_)) * np.sum(cov_, axis=0)

        return cov, cov_, centroids

    def _compute_scatter(self, X_, X):
        # Estimate class centroids and data centroid
        centroids = np.array([np.mean(x, axis=0) for x in X_])
        data_centroid = np.mean(X, axis=0)

        # Between-class and within-class scatter matrices
        S_b = np.sum([X_[i].shape[0] *  np.outer(c - data_centroid, c-data_centroid) for i, c in enumerate(centroids)], axis=0)
        S_w = np.sum([(X_[i] - c).T.dot(X_[i] - c) for i, c in enumerate(centroids)], axis=0)

        return S_b, S_w

class LinearDiscriminantAnalysis(DiscriminantAnalysis):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        X_, self.class_priors, self.n_classes = super()._prepare_data(X, y)
        self.cov, _, self.centroids = super()._compute_covariances(X_)

        # Determine target dimensionality if not given
        if self.n_components is None:
            # No information loss when projecting into vector space spanned by class centroids
            self.n_components = min(self.n_classes - 1, X.shape[1])

        if self.n_components < X.shape[1]:
            # Compute scatter matrices and solve Fisher eigenproblem
            S_b, S_w = super()._compute_scatter(X_, X)
            w, V = linalg.eig(S_b, S_w)
            w, V = w.real, V.real

            # Take self.n_components eigenvectors with largest eigenvalues as new basis/projection matrix
            self.proj = V[(-w).argsort()][:self.n_components]

            # Compute mean and shared covariance of transformed class distributions
            self.cov = self.proj.dot(self.cov).dot(self.proj.T)
            self.centroids = self.centroids.dot(self.proj.T)
        else:
            self.proj = np.identity(X.shape[1])

    def transform(self, X):
        X_new = X.dot(self.proj.T)
        return X_new

    def predict_proba(self, X):
        # Project
        X_new = self.transform(X)

        # P(cls|X) = P(X|cls) * P(cls)/ P(X), with P(X)=sum_{c} P(X|c)
        posteriors = np.array([self.class_priors[cls] * multivariate_normal.pdf(X_new, self.centroids[cls], self.cov) for cls in range(self.n_classes)]).T
        prob = posteriors / np.repeat(np.sum(posteriors,axis=1)[...,np.newaxis], posteriors.shape[1], axis=1)
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

class QuadraticDiscriminantAnalysis(DiscriminantAnalysis):

    def fit(self, X, y):
        X_, self.class_priors, self.n_classes = super()._prepare_data(X, y)
        _, self.cov_, self.centroids = super()._compute_covariances(X_)

    def predict_proba(self, X):
        # P(cls|X) = P(X|cls) / P(X), with P(X)=sum_{c} P(X|c)
        posteriors = np.array([self.class_priors[cls] * multivariate_normal.pdf(X, self.centroids[cls], self.cov_[cls]) for cls in range(self.n_classes)]).T
        prob = posteriors / np.repeat(np.sum(posteriors,axis=1)[...,np.newaxis], posteriors.shape[1], axis=1)
        return prob

    def predict(self, X):
        prob = self.predict_proba
        return np.argmax(prob, axis=1)
