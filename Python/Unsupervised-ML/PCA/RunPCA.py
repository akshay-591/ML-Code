# this file contain method for PCA algorithm which will return the reduced dimension dataset

import numpy as mat


def execute(X):
    sigma = mat.divide(mat.dot(X.transpose(), X), X.shape[0])
    U, S, V = mat.linalg.svd(sigma)

    return U, S
