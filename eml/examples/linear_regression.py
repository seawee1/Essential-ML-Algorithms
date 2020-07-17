from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import spearmanr

from eml.supervised_learning.regression import LinearRegression

def main():
    # Load data
    X, y = load_boston(return_X_y=True)
    # Select variable with biggest (absolute) spearman correlation coefficient to target (just for visually appealing results)
    spear_coef = [spearmanr(X[:,i], y)[0] for i in range(X.shape[1])]
    X = np.expand_dims(X[:, np.argmax(np.abs(spear_coef))], axis=1)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1337)

    # Train model
    n_iter=10000
    model = LinearRegression(n_iter=n_iter, lr=0.001)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    mse = (1/X_test.shape[0]) * np.sum((y_test - y_pred)**2)

    # Plot learning curve
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.arange(n_iter), model.loss_history, color='black', linewidth=2)
    ax1.set_title("Linear Regression (Testset MSE: %.2f)" % mse, fontsize=12)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MSE")
    # Plot predictions
    cmap = plt.get_cmap('viridis')
    m1 = ax2.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = ax2.scatter(X_test, y_test, color=cmap(0.5), s=10)
    ax2.plot(X_test, y_pred, color='black', linewidth=2, label="Prediction")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
