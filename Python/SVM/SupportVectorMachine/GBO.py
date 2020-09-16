# this file contains gradient based optimization methods for SVM also known as Soft margin SVM

import numpy as mat


def cost(W, X, Y, C):
    # calculate linear hypothesis

    W = mat.c_[W]
    m = X.shape[0]
    fx = mat.dot(X, W)
    slack = mat.subtract(1, mat.multiply(Y, fx))

    slack = mat.sum(mat.where(slack <= 0, 0, slack))
    hinge_loss = C*slack

    margin = mat.multiply(1/2, mat.dot(W.transpose(), W))

    final_loss = mat.multiply(1 / m, mat.add(margin, hinge_loss))

    return final_loss


def grad(W, X, Y, C):
    W = mat.c_[W]
    m = X.shape[0]
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
    dw1 = mat.add(W, mat.multiply(C, -opt))

    # final
    dw = mat.add(dw0, dw1)

    return dw

