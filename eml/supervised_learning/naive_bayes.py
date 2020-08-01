import numpy as np
from scipy.stats import multivariate_normal

class GaussianNB():

    def fit(self, X, y):
        # Estimate class priors
        unique, unique_counts = np.unique(y, return_counts=True)
        self.n_classes = len(unique)
        self.class_priors = unique_counts / len(y)
        # Estimate normal distribution parameters (MLE)
        # Naive Bayes is equivalent to assuming Normal distributed classes with diagonal covariance matrix
        self.centroids = [np.mean(X[y == cls], axis=0) for cls in range(self.n_classes)]
        self.cov = [np.diag(np.mean((X[y==cls] - self.centroids[cls])**2, axis=0)) for cls in range(self.n_classes)]

    def predict_proba(self, X):
        posteriors = np.array([self.class_priors[cls] * multivariate_normal.pdf(X, self.centroids[cls], self.cov[cls]) for cls in range(self.n_classes)]).T
        prob = posteriors / np.repeat(np.sum(posteriors,axis=1)[...,np.newaxis], posteriors.shape[1], axis=1)
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)