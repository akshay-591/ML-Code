# this file contains code for regularised cos and gradient

import numpy as mat
from scipy import optimize


def hypo(x, theta):
    # calculating prediction
    if theta.shape[0] != x.shape[1] and theta.shape[1] == x.shape[1]:
        prediction = mat.dot(x, theta.transpose())
    else:
        prediction = mat.dot(x, theta)
    return prediction


def cal_cost(theta, x, y, lamb):
    example_size = x.shape[0]
    # theta array needs to be flatten since sciipy optimization function wont work well with arrays
    # so we need to convert them in list
    theta = theta.flatten()
    theta = mat.c_[theta]
    # calculating prediction
    pred = hypo(x, theta)
    value = (1 / (2 * example_size))
    # calculating error
    error = mat.subtract(pred, mat.c_[y])

    # calculating square error
    square_error = mat.dot(error.transpose(), error)
    # final calculation
    j = mat.multiply(square_error, value)

    regularized = mat.dot((lamb / (2 * example_size)),
                          mat.dot(theta[1:theta.shape[0], :].transpose(), theta[1:theta.shape[0], :]))
    j = j + regularized
    return j


def cal_grad(theta, X, y, lamb):
    total_example = X.shape[0]
    theta = theta.flatten()
    initial_theta = mat.c_[theta]
    # calculate prediction
    prediction = hypo(X, initial_theta)
    opt_theta = mat.multiply(1 / total_example, mat.dot(X.transpose(), mat.subtract(prediction, y)))
    # calculating grads
    # regularising
    reg = mat.multiply((lamb / total_example), initial_theta[1:initial_theta.shape[0]])
    reg = mat.add(opt_theta[1:opt_theta.shape[0], :], reg)
    regularised_Para = mat.c_[opt_theta[0, :], reg.transpose()].transpose()
    return regularised_Para.flatten()


def optimize_grad(X, Y, lamb):
    initial_theta = mat.zeros(X.shape[1])
    result = optimize.minimize(fun=cal_cost,
                               x0=initial_theta,
                               args=(X, Y, lamb),
                               method='CG',
                               jac=cal_grad,
                               options={'maxiter':200})
    return result.x