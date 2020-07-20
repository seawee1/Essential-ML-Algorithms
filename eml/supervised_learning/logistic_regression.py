import numpy as np

class LogisticRegression():
    def __init__(self, n_iter=100, lr=0.001):
        self.n_iter = n_iter
        self.lr = lr
        self.learning_curve = []

    def init_weights(self, n_weights):
        self.w = np.zeros(n_weights)

    def sigmoid(self, X, w):
        return 1/(1+np.exp(-w.dot(X.T)))

    def fit(self, X, y):
        # Append ones column for easier handling of intercept term
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.init_weights(X.shape[1])

        for i in range(self.n_iter):
            # Calculate P(Y=1|X=x)
            prob = self.sigmoid(X, self.w)

            # Negative Log-likelihood
            prob_ = np.copy(prob)
            prob_[y==0] = 1.0 - prob_[y==0]
            nll = -np.log(np.sum(prob_))
            self.learning_curve.append(nll)

            # Gradient of neg. log-likelihood w.r.t slope and intercept
            grad_w = -(y - prob).dot(X)

            # Gradient descent
            self.w = self.w - (grad_w * self.lr)
    def predict(self, X):
        # Calculate sigmoid
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        prob = self.sigmoid(X, self.w)

        # Threshold into classes
        y_pred = np.zeros((X.shape[0],))
        y_pred[prob>=0.5] = 1
        return y_pred

    def predict_proba(self, X):
        # Calculate sigmoid
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        prob = np.expand_dims(self.sigmoid(X, self.w), axis=1)
        ret = np.hstack((1-prob, prob))
        return ret
