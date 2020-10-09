"""
This file have a method which will normalize the data
"""
import numpy as mat


def normalizeData(Y, R):
    Ynorm = mat.zeros(Y.shape)
    Ymean = mat.zeros((len(Y), 1))
    # first get the indices from all rows of R matrix where value is = 1.

    for i in range(len(Y)):
        index = mat.where(R[i, :] == 1)[0]
        Ymean[i] = mat.mean(Y[i, index])
        Ynorm[i, index] = mat.subtract(Y[i, index], Ymean[i])

    return Ymean, Ynorm

