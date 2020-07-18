from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal

from eml.supervised_learning.logistic_regression import LogisticRegression

def main():
    # Load data
    #X, y = load_breast_cancer(return_X_y=True)
    # Split data

    X_0 = multivariate_normal([-1, 1], [[1, 0],[0,1]], 500)
    X_1 = multivariate_normal([1, -1], [[1, 0],[0,1]], 500)
    X = np.vstack((X_0, X_1))
    #plt.scatter(X_0[:,0].flatten(), X_0[:,1].flatten())
    #plt.scatter(X_1[:,0].flatten(), X_1[:,1].flatten())
    #plt.show()
    y = np.zeros((1000,))
    y[500:] = 1

    idx = np.arange(1000)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1337)


    # Train model
    n_iter=10000
    lr = 0.001
    model = LogisticRegression(n_iter=n_iter, lr=0.001)
    model.fit(X, y)

    # X - some data in 2dimensional np.array

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                              np.arange(y_min, y_max, 0.1))
    # here "model" is your model's prediction (classification) function
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.axis('off')

    # Plot also the training points
    cmap = plt.get_cmap('viridis')
    m1 = plt.scatter(X[y==0][:,0], X[y==0][:,1], color=cmap(0.9), s=10)
    m2 = plt.scatter(X[y==1][:,0], X[y==1][:,1], color=cmap(0.1), s=10)
    #plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    plt.show()


if __name__=='__main__':
    main()
