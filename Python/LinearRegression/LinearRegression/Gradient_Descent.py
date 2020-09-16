# this file is for calculating Gradient Descent

import numpy as mat
import Linear_hypothesis


def cal_grad(x, y, theta, alpha, iteration):
    example_size = x.shape[0]
    # calculating prediction
    for i in range(1, iteration):
        pred = Linear_hypothesis.hypo(x, theta)
        value = (alpha / example_size)
        # calculating error
        error = mat.subtract(pred, mat.c_[y])
        # optimization
        mini_j = mat.dot(error.transpose(), mat.c_[mat.ones(example_size), x])
        opt = (mat.multiply(mini_j, value)).transpose()

        theta = mat.subtract(theta, opt)

    return theta
