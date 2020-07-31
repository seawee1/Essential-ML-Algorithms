import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles

from eml.supervised_learning.logistic_regression import LogisticRegression

import matplotlib.pyplot as plt
import eml.utils.plot as plot

n_iter=1000
lr = 0.001
n_samples = 300

models = [
    LogisticRegression(n_iter, lr)]
names = [
    'LogisticRegression']
datasets = [
    (make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=1.2)),
    (make_moons(n_samples=n_samples, noise=0.3)),
    (make_circles(n_samples=n_samples, noise=0.2, factor=0.5))
]

fig = plt.figure(figsize=(len(datasets)*5, (len(models) + 1) * 5))

for i, (X, y) in enumerate(datasets):
    plt_idx = i*(len(models)+1) + 1
    # Plot data
    ax = fig.add_subplot(len(datasets), len(models) + 1, plt_idx)
    print('outer', i*(len(models)+1)+1)
    plot.scatter_2d(X, y, ax)
    if i == 0:
        ax.set_title("Input data")

    # Fit models to datasets[i] and plot results
    for j, model in enumerate(models):
        model.fit(X, y)
        #y_pred = model.predict_proba(X)
        ax = fig.add_subplot(len(datasets), len(models) + 1, plt_idx+j+1)
        plot.decision_regions_2d(X, y, model, ax)
        if i == 0:
            ax.set_title(names[j])

plt.show()
