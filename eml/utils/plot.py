import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

cm = ListedColormap(['#FF0000', '#0000FF'])
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def scatter_2d(X, y, ax=None):
    ax = plt.gca() if ax is None else ax

    if X.shape[1] == 1:
        ax.scatter(X[:,0], np.zeros(X.shape[0]), c=y, cmap=cm, s=10, marker='o', edgecolor='black')
    else:
        ax.scatter(X[:,0], X[:,1], c=y, cmap=cm, s=10, marker='o', edgecolor='black')

def scatter_3d(X, y, ax):
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap=cm, s=10, marker='o', edgecolor='black')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

def decision_regions_2d(X, y, model, ax=None, use_proba=True, resolution=50):
    # Colormaps
    cm = plt.cm.RdBu
    ax = ax or plt.gca()

    # Create meshgrid to predict
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max - x_min) / resolution
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict either class labels or probabilities
    if use_proba:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Draw decision regions
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.7)

    # Plot training data
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, s=10, marker='o', edgecolor='black')
