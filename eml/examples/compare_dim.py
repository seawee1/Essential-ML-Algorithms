import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import sklearn.discriminant_analysis as da

import eml.utils.datasets as datasets
import eml.utils.plot as plot
from eml.supervised_learning.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def main():
    X_easy, y_easy = datasets.binary_gaussian()
    X_digits, y_digits = load_digits(return_X_y=True)
    print(X_digits[0])

    '''
    # Fit model on easy data
    model = LinearDiscriminantAnalysis(n_components=1)
    model.fit(X_easy, y_easy)
    model_sk = da.LinearDiscriminantAnalysis(n_components=1, solver='eigen')
    model_sk.fit(X_easy, y_easy)

    plot.data(X_easy, y_easy)
    plt.show()
    fig, axs = plt.subplots(2)
    plot.data(model.transform(X_easy), y_easy, ax=axs[0])
    plot.data(model_sk.transform(X_easy), y_easy, ax=axs[1])
    plt.show()
    '''

    # The same for hard data
    print(X_digits.shape)
    model = LinearDiscriminantAnalysis(n_components=4)
    model.fit(X_digits, y_digits)
    model_sk = da.LinearDiscriminantAnalysis(n_components=4, solver='eigen')
    model_sk.fit(X_digits, y_digits)

    #plot.data(X_easy, y_easy)
    #plt.show()
    X_new = model.transform(X_digits)
    X_new_sk = model_sk.transform(X_digits)
    fig, axs = plt.subplots(2,2)
    plot.data(X_new[:,0:2], ax=axs[0,0])
    plot.data(X_new_sk[:,0:2], ax=axs[0,1])
    plot.data(X_new[:,2:4], ax=axs[1,0])
    plot.data(X_new_sk[:,2:4], ax=axs[1,1])
    plot.data(model_sk.transform(X_digits), y_digits, ax=axs[1])
    plt.show()

if __name__=='__main__':
    main()
