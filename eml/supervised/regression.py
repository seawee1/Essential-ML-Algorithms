import numpy as np

class LinearRegression():
    def __init__(self, n_iter=100, lr=0.001):
        self.n_iter = n_iter
        self.lr = lr

    def init_weights(self, n_weights):
        self.w = np.zeros(n_weights)

    def fit(self, X, y):
        # Append ones column for easier handling of intercept term
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.init_weights(X.shape[1])
        self.loss_history = []

        for i in range(self.n_iter):
            y_pred = X.dot(self.w)
            mse = (1.0/X.shape[0]) * np.sum((y - y_pred)**2)
            # Compute gradient w.r.t slope and intercept
            grad_w = np.multiply(-X, (2*(y - y_pred)).reshape((-1,1)))
            grad_w = (1.0/X.shape[0]) * np.sum(grad_w, axis=0)

            # Update weights
            self.w = self.w - (grad_w * self.lr)
            self.loss_history.append(mse)

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        y_pred = X.dot(self.w)
        return y_pred
