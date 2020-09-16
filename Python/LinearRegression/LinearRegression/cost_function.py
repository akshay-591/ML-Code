# this file contains the cost function for the program

import numpy as mat
import Linear_hypothesis


def cal_cost(x, y, theta):

    example_size = x.shape[0]
    # calculating prediction
    pred = Linear_hypothesis.hypo(x, theta)
    value = (1 / (2 * example_size))
    # calculating error
    error = mat.subtract(pred, mat.c_[y])

    # calculating square error
    square_error = mat.dot(error.transpose(), error)
    # final calculation
    j = mat.multiply(square_error, value)
    return float(j)
