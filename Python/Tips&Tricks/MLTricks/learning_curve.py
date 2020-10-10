# this file contains method which will return the set of errors of training set and cross validation set

import numpy as mat

from MLTricks import Regularised_Linear_Cost_Grad as Rg


def curve(X, Y, Xval, Yval, lamb):
    error_train = mat.zeros((X.shape[0], 1))
    error_Val = mat.zeros((X.shape[0], 1))
    iteration = mat.c_[mat.arange(start=0, stop=X.shape[0], step=1)]

    for i in range((len(Y))):
        theta = Rg.optimize_grad(X[0:i + 1, :], Y[0:i + 1], lamb)

        error_train[i, :] = Rg.cal_cost(theta, X[0:i + 1, :], Y[0:i + 1, :], 0)
        error_Val[i, :] = Rg.cal_cost(theta, Xval, Yval, 0)

    return error_train, error_Val, iteration
