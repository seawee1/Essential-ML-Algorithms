import numpy as np
from sklearn.datasets import make_classification

from eml.supervised_learning.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import eml.utils.plot as plot

n_samples = 300

models = [
    LinearDiscriminantAnalysis(n_components=2)]
names = [
    'LDA']
datasets = [
    (make_classification(n_samples=300, n_features=3, n_redundant=0, n_informative=3, n_clusters_per_class=1, class_sep=2.0))]


fig = plt.figure(figsize=((len(models) + 1) * 5, len(datasets)*5))

for i, (X, y) in enumerate(datasets):
    # Plot data
    plt_idx = i*(len(models)+1) + 1
    ax = fig.add_subplot(len(datasets), len(models) + 1, plt_idx, projection='3d')
    plot.scatter_3d(X, y, ax)
    if i == 0:
        ax.set_title("Input data")

    # Fit models to datasets[i] and plot results
    for j, model in enumerate(models):
        model.fit(X, y)
        X_new = model.transform(X)
        ax = fig.add_subplot(len(datasets), len(models) + 1, plt_idx+j+1)
        plot.scatter_2d(X_new, y, ax)
        if i == 0:
            ax.set_title(names[j])

plt.savefig('img/dim_reduction.png')
plt.show()

