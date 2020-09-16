# this file is for calculating Gradient Descent

import numpy as mat
import Linear_hypothesis


def cost(theta, x, y):
    example_size = x.shape[0]

    initial_theta = mat.c_[theta]

    # calculating prediction
    pred = Linear_hypothesis.hypo(x, initial_theta)
    value = (1 / (2 * example_size))

    # calculating error
    error = mat.subtract(pred, mat.c_[y])

    # calculating square error
    square_error = mat.dot(error.transpose(), error)
    # final calculation
    j = mat.dot(square_error, value)

    return float(j)


def cal_grad(theta, x, y):
    example_size = x.shape[0]
    initial_theta = mat.c_[theta]


    # calculating prediction
    pred = Linear_hypothesis.hypo(x, initial_theta)

    value = (1 / example_size)
    # calculating error
    error = mat.subtract(pred, mat.c_[y])
    # optimization
    mini_j = mat.dot(error.transpose(), mat.c_[mat.ones(example_size), x])
    opt = mat.dot(mini_j, value)

    return opt.transpose()


def cost_grad(theta, x, y):
    example_size = x.shape[0]
    n = theta.shape[0]

    initial_theta = mat.zeros([1, n])

    # calculating prediction
    pred = Linear_hypothesis.hypo(x, initial_theta)
    value = (1 / (2 * example_size))

    # calculating error
    error = mat.subtract(pred, mat.c_[y])

    # calculating square error
    square_error = mat.dot(error.transpose(), error)
    # final calculation
    j = mat.dot(square_error, value)
    val = 1/example_size
    mini_j = mat.dot(error.transpose(), mat.c_[mat.ones(example_size), x])
    opt = mat.multiply(mini_j, val)
    return j, opt


