# This file Contains Method which will find out the Principal Components from best to worst for the Data matrix

import numpy as mat


def execute(X):
    sigma = mat.divide(mat.dot(X.transpose(), X), X.shape[0])
    U, S, V = mat.linalg.svd(sigma)

    return U, S
