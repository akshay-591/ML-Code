"""
This file contains code for checking if cost function and Gradient function we have implemented working
Correctly rather than applying directly to the data we will use some small random generated data.
it makes easier to check if functions are flexible or not
"""

import numpy as mat
from Collaborativefiltering import CostGrad
from DebuggingTool import testNumericalGradient


def checkGrads(lamb):
    # create small data
    X_t = mat.random.rand(4, 3)  # to create Y and R
    theta_t = mat.random.rand(5, 3)  # to create Y and R

    # remove most entries
    Y = mat.dot(X_t, theta_t.transpose())
    Y[mat.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = mat.where(Y == 0, 0, 1)
    # Run checker
    X = mat.random.randn(4, 3)
    Theta = mat.random.randn(5, 3)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_feature = X_t.shape[1]

    param = mat.r_[X.flatten(), Theta.flatten()]

    # calculate numerical gradient
    test_grad = testNumericalGradient.NumGrad(function=CostGrad.cal_cost,
                                              theta=param,
                                              parameters=(Y, R, num_feature, num_movies, num_users, lamb))

    # calculate analytical gradient
    analytical_gard = CostGrad.cal_grad(param, Y, R, num_feature, num_movies, num_users, lamb)

    # calculate difference
    mat_a = mat.subtract(test_grad, analytical_gard)
    mat_b = mat.add(test_grad, analytical_gard)
    # calculate norm
    diff = mat.linalg.norm(mat_a) / mat.linalg.norm(mat_b)

    # diff = "{:e}".format(diff)
    return test_grad, analytical_gard, diff
