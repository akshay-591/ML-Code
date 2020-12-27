"""
This file contain methods for calculating prediction, Cost and Gradient Descent.
"""
import numpy as mat


def hypo(x, theta):
    """
    This method calculate prediction for the Model
    :param x: input parameters
    :param theta: weight parameters
    :return: prediction
    """
    # calculating prediction
    if theta.shape[0] != x.shape[1] and theta.shape[1] == x.shape[1]:
        prediction = mat.dot(x, theta.transpose())
    else:
        prediction = mat.dot(x, theta)
    return prediction


def cal_cost(theta, x, y):
    """
    This method is for Calculating Error between prediction and original output using Least Mean Square technique.

    :param theta: weight vectors.
    :param x: input matrix
    :param y: output matrix
    :return: error or loss
    """
    example_size = x.shape[0]

    theta = mat.c_[theta]
    # calculating prediction
    pred = hypo(x, theta)

    # calculating error
    error = mat.subtract(pred, mat.c_[y])

    # calculating square error
    square_error = mat.dot(error.transpose(), error)

    # final calculation
    j = mat.multiply(square_error, (1 / (2 * example_size)))
    return j


def cal_grad(x, y, theta, alpha, iteration):
    """
    This method of Gradient Descent is used when user do not uses built in optimization methods.

    :param x: input matrix
    :param y: output matrix
    :param theta: weight vectors
    :param alpha: step value
    :param iteration: number of iteration
    :return: optimized weight vectors
    """
    example_size = x.shape[0]
    theta = mat.c_[theta]
    # calculating prediction
    for i in range(1, iteration):
        pred = hypo(x, theta)
        value = (alpha / example_size)
        # calculating error
        error = mat.subtract(pred, mat.c_[y])
        # optimization
        mini_j = mat.dot(error.transpose(), x)
        opt = (mat.multiply(mini_j, value)).transpose()

        theta = mat.subtract(theta, opt)

    return theta


def grad_opt(theta, x, y):
    """
    This Method is for Gradient Descent and only usable with built in optimization method like Conjugate Gradient etc
    :param theta: Weight Vectors
    :param x: input matrix
    :param y: output matrix
    :return: derivative of function (weight Vectors)
    """

    example_size = x.shape[0]
    theta = mat.c_[theta]
    pred = hypo(x, theta)
    value = (1 / example_size)
    # calculating error
    error = mat.subtract(pred, mat.c_[y])
    # optimization
    mini_j = mat.dot(error.transpose(), x)
    opt = (mat.multiply(mini_j, value)).transpose()

    return opt.flatten()
