# this is file for Linear Hypothesis

import numpy as matrix

# a = matrix.array([1, 2, 3])
# b = matrix.zeros((1, 3))

# print(b.transpose())


def hypo(x, theta):
    row_size = x.shape[0]
    x = matrix.c_[matrix.ones(row_size), x]
    # calculating prediction
    prediction = matrix.dot(x, theta)
    return prediction

