import matplotlib.pyplot as plt

import eml.utils.datasets as datasets
import eml.utils.plot as plot
from eml.supervised_learning.logistic_regression import LogisticRegression

def main():
    # Load data
    X, y = datasets.binary_classification()

    # Train model
    n_iter=10000
    lr = 0.001
    model = LogisticRegression(n_iter=n_iter, lr=lr)
    model.fit(X, y)

    # Show results
    plot.decision_regions(X, y, model)
    plt.show()

if __name__=='__main__':
    main()
