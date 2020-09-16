# this file contain loss function for logistic regression

import numpy as mat
from LogisticRegression import sigmoidFuntion


def lossFun(theta, X, Y):
    total_example = X.shape[0]
    initial_theta = mat.c_[theta]
    # calculate prediction
    prediction = sigmoidFuntion.hypo(initial_theta, X)
    sigm = sigmoidFuntion.sigmoid(prediction)

    # Loss when when Y=1
    loss0 = mat.dot(-Y.transpose(), mat.log(sigm))

    # loss when Y=0

    loss1 = mat.dot(mat.subtract(1, Y).transpose(), mat.log(mat.subtract(1, sigm)))

    # Total Avg loss

    loss_final = mat.multiply((1 / total_example), mat.subtract(loss0, loss1))

    return loss_final


def regularised_cost(theta, X, Y, lamb):
    total_example = X.shape[0]
    # calculate cost
    loss = lossFun(theta, X, Y)
    theta = mat.c_[theta]
    # regularize parameter = 1/2m * sum(theta(i)^2) from i=1 to n where n is number of features

    regularized = mat.dot(lamb / (2 * total_example),
                          mat.dot(theta[1:theta.shape[0], :].transpose(), theta[1:theta.shape[0], :]))

    return mat.add(loss, regularized)


def grad(theta, X, Y):
    total_example = X.shape[0]
    initial_theta = mat.c_[theta]
    # calculate prediction
    prediction = sigmoidFuntion.hypo(initial_theta, X)
    sigm = sigmoidFuntion.sigmoid(prediction)

    opt_theta = mat.multiply(1 / total_example, mat.dot(X.transpose(), mat.subtract(sigm, Y)))

    return opt_theta


def regularised_grad(theta, X, Y, lamb):
    total_example = X.shape[0]
    # calculating grads
    optimum_grad = grad(theta, X, Y)
    theta = mat.c_[theta]
    # regularising
    reg = mat.multiply(lamb / total_example, theta[1:theta.shape[0]])
    reg = mat.add(optimum_grad[1:optimum_grad.shape[0], :], reg)

    regularised_Para = mat.c_[optimum_grad[0, :], reg.transpose()]

    return regularised_Para.transpose()
