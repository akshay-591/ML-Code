"""
This file Contains Methods for Calculating Cost/Loss and Optimizing weight Vectors
"""

import numpy as mat
from LogisticRegression import sigmoidFuntion


def regularised_cost(theta, X, Y, lamb):
    """
    This method Calculates the error difference using Maximum Likelihood Technique and also add Regularisation.

    :param theta: weight vectors
    :param X: input matrix
    :param Y: output matrix
    :param lamb: Regularization parameter
    :return: error/cost
    """
    total_example = X.shape[0]
    theta = mat.c_[theta]

    # calculate prediction
    prediction = sigmoidFuntion.hypo(theta, X)
    sigm = sigmoidFuntion.sigmoid(prediction)

    # Loss when when Y=1
    loss0 = mat.dot(-Y.transpose(), mat.log(sigm))

    # loss when Y=0

    loss1 = mat.dot(mat.subtract(1, Y).transpose(), mat.log(mat.subtract(1, sigm)))

    # Total Avg loss

    loss_final = mat.multiply((1 / total_example), mat.subtract(loss0, loss1))

    # calculate cost

    # regularize parameter = 1/2m * sum(theta(i)^2) from i=1 to n where n is number of features

    regularized = mat.dot(lamb / (2 * total_example),
                          mat.dot(theta[1:theta.shape[0], :].transpose(), theta[1:theta.shape[0], :]))

    return mat.add(loss_final, regularized)


def regularised_grad(theta, X, Y, lamb):
    """
    This method Calculates the derivative of the loss function.

    :param theta: Weight vectors
    :param X: input matrix
    :param Y: output matrix
    :param lamb: Regularization parameters
    :return: derived value (or slop of the Tangent Line to the function)
    """
    total_example = X.shape[0]
    theta = mat.c_[theta]
    # calculate prediction
    prediction = sigmoidFuntion.hypo(theta, X)
    sigm = sigmoidFuntion.sigmoid(prediction)

    optimum_grad = mat.multiply(1 / total_example, mat.dot(X.transpose(), mat.subtract(sigm, Y)))

    # regularising
    reg = mat.multiply(lamb / total_example, theta[1:theta.shape[0]])
    reg = mat.add(optimum_grad[1:optimum_grad.shape[0], :], reg)

    regularised_Para = mat.c_[optimum_grad[0, :], reg.transpose()]

    return regularised_Para.transpose()
