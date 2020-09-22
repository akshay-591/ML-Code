# This file contains method which will project the original data points on Principal component or subspace

import numpy as mat


def project(X, U, numPCs):
    Z = mat.dot(X, U[:, 0:numPCs])
    return Z
