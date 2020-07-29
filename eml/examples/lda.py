import matplotlib.pyplot as plt

import eml.utils.datasets as datasets
import eml.utils.plot as plot
from eml.supervised_learning.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def main():
    X, y = datasets.binary_gaussian()
    #plot.data(X, y)
    #plt.show()

    # Train model
    n_iter=10000
    lr = 0.001
    model = LinearDiscriminantAnalysis(n_components=1)
    #model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)
    X_new = model.transform(X)

    plot.data(X, y)
    plt.show()
    plot.data(X_new, y)
    plt.show()

    # Show results
    #plot.decision_regions(X, y, model, use_proba=False)
    #plt.show()

if __name__=='__main__':
    main()
