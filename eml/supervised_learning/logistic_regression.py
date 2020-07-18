import numpy as np

class LogisticRegression():
    def __init__(self, n_iter=100, lr=0.001):
        self.n_iter = n_iter
        self.lr = lr
        self.learning_curve = []

    def init_weights(self, n_weights):
        self.w = np.zeros(n_weights)

    def fit(self, X, y):
        # Append ones column for easier handling of intercept term
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.init_weights(X.shape[1])

        for i in range(self.n_iter):
            # Regression and sigmoid activation
            reg = self.w.dot(X.T)
            pred = 1/(1+np.exp(-reg))
            #print(pred)

            # Log-likelihood
            #llh = np.sum(np.log(np.exp(1 - pred)) +  #np.multiply(y, reg))
            #llh = np.sum(np.multiply(y, np.log(np.log(np.exp(pred)))) + np.multiply(1-y, np.log(np.log(np.exp(1-pred))))
            pred_ = np.copy(pred)
            pred_[y==0] = 1.0 - pred_[y==0]
            #print(pred)
            #llh = np.sum(np.log(np.log(np.exp(pred_))))
            #self.learning_curve.append(llh)
            nll = -np.log(np.sum(pred_))
            print(nll)

            # Gradient of neg. log-likelihood w.r.t slope and intercept
            grad_w = -(y - pred).dot(X)

            self.w = self.w - (grad_w * self.lr)

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        reg = self.w.dot(X.T)
        y_pred = 1/(1+np.exp(-reg))
        return y_pred
