# this file contains method which will return the set of errors of training set and cross validation set

import numpy as mat
from scipy import optimize

from MLTricks import Regularised_Linear_Cost_Grad as Rg


def curve(X, Y, Xval, Yval, lamb):
    X = mat.c_[mat.ones(X.shape[0]), X]
    Xval = mat.c_[mat.ones(Xval.shape[0]), Xval]
    error_train = mat.zeros((X.shape[0], 1))
    error_Val = mat.zeros((X.shape[0], 1))
    iteration = mat.c_[mat.arange(start=0, stop=X.shape[0], step=1)]
    for i in range((len(Y))):
        initial_theta = mat.zeros(X.shape[1])
        result = optimize.minimize(fun=Rg.cal_cost,
                                   x0=initial_theta,
                                   args=(X[0:i + 1, :], Y[0:i + 1], lamb),
                                   method='CG',
                                   jac=Rg.cal_grad,
                                   options={'maxiter': 200})
        theta = result.x
        error_train[i, :] = Rg.cal_cost(theta, X[0:i + 1, :], Y[0:i + 1, :], 0)
        error_Val[i, :] = Rg.cal_cost(theta, Xval, Yval, 0)

    return error_train, error_Val, iteration


def curvetemp(X, Y, Xval, Yval, lamb):
    X = mat.c_[mat.ones(X.shape[0]), X]
    Xval = mat.c_[mat.ones(Xval.shape[0]), Xval]
    error_train = mat.zeros((X.shape[0], 1))
    error_Val = mat.zeros((X.shape[0], 1))
    iteration = mat.c_[mat.arange(start=0, stop=X.shape[0], step=1)]
    for i in range((len(Y))):
        initial_theta = mat.zeros(X.shape[1])
        result = optimize.minimize(fun=Rg.cal_cost,
                                   x0=initial_theta,
                                   args=(X[0:i + 1, :], Y[0:i + 1], lamb),
                                   method='Powell')
        theta = result.x
        theta = theta.transpose()
        error_train[i, :] = Rg.cal_cost(theta, X[0:i + 1, :], Y[0:i + 1, :], 0)
        error_Val[i, :] = Rg.cal_cost(theta, Xval, Yval, 0)

    return error_train, error_Val, iteration
