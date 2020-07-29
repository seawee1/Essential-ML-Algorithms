import numpy as np
import sklearn.discriminant_analysis as da
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from eml.supervised_learning.discriminant_analysis import LinearDiscriminantAnalysis
import eml.utils.plot as plot
import eml.utils.datasets as datasets

def main():
    '''
    # 2D data
    X_2d, y_2d = datasets.binary_gaussian()
    model = LinearDiscriminantAnalysis(n_components=1)
    model.fit(X_2d, y_2d)
    model_sk = da.LinearDiscriminantAnalysis(n_components=1, solver='eigen')
    model_sk.fit(X_2d, y_2d)

    plot.data(X_2d, y_2d)
    plt.show()
    fig, axs = plt.subplots(2)
    plot.data(model.transform(X_2d), y_2d, ax=axs[0])
    plot.data(model_sk.transform(X_2d), y_2d, ax=axs[1])
    plt.show()
    '''

    # 3D data
    model = LinearDiscriminantAnalysis(n_components=2)
    model.fit(X_3d, y_3d)
    model_sk = da.LinearDiscriminantAnalysis(n_components=2, solver='eigen')
    model_sk.fit(X_3d, y_3d)

    X_3d, y_3d = datasets.multi_gaussian_3d()
    fig = plt.figure()
    ax = Axes3D(fig)
    plot.data_3d(X_3d, y_3d, ax)
    plt.show()

    X_new = model.transform(X_3d)
    X_new_sk = model_sk.transform(X_3d)
    fig, axs = plt.subplots(2)
    plot.data(X_new, y_3d, ax=axs[0])
    plot.data(X_new_sk, y_3d, ax=axs[1])
    plt.show()

if __name__=='__main__':
    main()
