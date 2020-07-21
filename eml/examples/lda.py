import matplotlib.pyplot as plt

import eml.utils.datasets as datasets
import eml.utils.plot as plot
from eml.supervised_learning.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    X, y = datasets.classification_gaussians()
    #plot.data(X, y)
    #plt.show()

    # Train model
    n_iter=10000
    lr = 0.001
    model = LinearDiscriminantAnalysis(n_iter=n_iter, lr=lr)
    model.fit(X, y)

    # Show results
    plot.decision_regions(X, y, model, use_proba=False)
    plt.show()

if __name__=='__main__':
    main()
