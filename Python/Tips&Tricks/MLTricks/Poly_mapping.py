# this method contains method for mapping features to higher dimension n

import numpy as mat


def mapPoly(X, power):
    X_mapped = mat.zeros((X.shape[0], power))
    j = 1
    for i in range(power):
        p= j+i

        X_mapped[:, i] = mat.array(mat.power(X, p)).flatten()

    return X_mapped