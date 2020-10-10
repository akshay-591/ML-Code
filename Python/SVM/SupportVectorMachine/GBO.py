"""
this file contains gradient based optimization methods for SVM also known as Soft margin SVM.
"""

import numpy as mat


def cost(W, X, Y, C):
    """
    This method calculate the Error/cost using Soft margin Technique.
    :param W: Weight Vectors including b
    :param X: input matrix
    :param Y: output matrix
    :param C: Trade of parameter
    :return: cost/error
    """
    # calculate linear hypothesis
    W = mat.c_[W]
    m = X.shape[0]

    fx = mat.dot(X, W)
    slack = mat.subtract(1, mat.multiply(Y, fx))

    # where slack value is less than equal to 0 replace that value with 0 and rest of the value remain same
    slack = mat.sum(mat.where(slack <= 0, 0, slack))
    # multiply with Trade of parameter
    hinge_loss = C*slack

    margin = mat.multiply(1/2, mat.dot(W.transpose(), W))

    final_loss = mat.multiply(1 / m, mat.add(margin, hinge_loss))

    return final_loss


def grad(W, X, Y, C):
    """
    This method is used for Optimization.

    :param W: Weight Vectors.
    :param X: Input matrix
    :param Y: Output matrix
    :param C: Trade-off parameter
    :return: derivative of loss function
    """
    W = mat.c_[W]
    fx = mat.dot(X, W)
    slack = mat.subtract(1, mat.multiply(Y, fx))

    # index where slack is less than 0
    slack_index0 = mat.where(slack <= 0)[0]

    # index where slack is more than 0
    slack_index1 = mat.where(slack > 0)[0]

    # optimization when slack is less than 0
    dw0 = mat.multiply(len(slack_index0), W)

    # optimization when slack is not less than 0
    X = X[mat.ix_(slack_index1)]
    Y = Y[mat.ix_(slack_index1)]

    opt = mat.dot(X.transpose(), Y)
    dw1 = mat.add(W, mat.dot(C, -opt))

    # final
    dw = mat.add(dw0, dw1)

    return dw

