# this file contain sigmoid function

import numpy as mat


def hypo(theta, X):
    # calculating prediction
    hypo = mat.dot(X, theta)
    return hypo


def sigmoid(z):
    sigm = 1 / (1 + mat.exp(-z))
    return sigm


