# this file contains code for regularised cos and gradient

import numpy as mat


def hypo(x, theta):
    # calculating prediction
    prediction = mat.dot(x, theta)
    return prediction


def cal_cost(theta, x, y, lamb):
    example_size = x.shape[0]
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

    regularized = mat.dot((lamb * (1 / (2 * example_size))),
                          mat.dot(theta[1:theta.shape[0], :].transpose(), theta[1:theta.shape[0], :]))
    j = j + regularized
    return j


def cal_grad(theta, X, y, lamb):
    total_example = X.shape[0]
    initial_theta = mat.c_[theta]
    # calculate prediction
    prediction = hypo(X, initial_theta)
    opt_theta = mat.multiply(1 / total_example, mat.dot(X.transpose(), mat.subtract(prediction, y)))
    # calculating grads
    # regularising
    reg = mat.multiply((lamb * (1 / total_example)), initial_theta[1:initial_theta.shape[0]])
    reg = mat.add(opt_theta[1:opt_theta.shape[0], :], reg)
    regularised_Para = mat.c_[opt_theta[0, :], reg.transpose()].transpose()
    return regularised_Para.flatten()
