# this file contain method to find out standrd deviation of matrix
import numpy as mat


# dimension =1 calculate standard deviation with 1/N
# and dimension = 0 calculate with 1/N-1
def stnd(X, dim):
    global result
    mean_norm = mat.mean(X, axis=0)

    if dim == 1:
        result = mat.sqrt(mat.multiply((1 / X.shape[0]), mat.sum(mat.power(mat.subtract(X, mean_norm), 2), axis=0)))
    if dim == 0:
        result = mat.sqrt(
            mat.multiply((1 / (X.shape[0] - 1)), mat.sum(mat.power(mat.subtract(X, mean_norm), 2), axis=0)))

    return result
