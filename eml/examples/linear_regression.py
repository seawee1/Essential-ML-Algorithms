from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

from ..supervised.regression import LinearRegression

X, y = load_boston(return_X_y=True)
model = LinearRegression(n_iter=500, lr=0.0000001)
model.fit(X, y)
