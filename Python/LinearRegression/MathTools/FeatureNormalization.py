# this file contains method for normalizing or scaling the features

import numpy as mat
from MathTools import StandardDeviation


def normalize(X):
    mu = mat.mean(X, axis=0)
    X_norm = mat.subtract(X, mu)
    sig = StandardDeviation.stnd(X_norm, dim=1)
    X_norm = mat.divide(X_norm, sig)

    return X_norm, mu, sig


def normalize2(X, mu, sig):
    X_norm = mat.subtract(X, mu)
    X_norm = mat.divide(X_norm, sig)

    return X_norm
