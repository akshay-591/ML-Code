# this method contains the metod for mapping features to higher dimension n

import numpy as mat


def map(X, power):
    X_mapped = mat.zeros((X.shape[0], power))
    j = 1
    for i in range(power):
        p= j+i
        #print(p)
        #print(mat.array(mat.power(X, p)).flatten())
        X_mapped[:, i] = mat.array(mat.power(X, p)).flatten()

    return X_mapped