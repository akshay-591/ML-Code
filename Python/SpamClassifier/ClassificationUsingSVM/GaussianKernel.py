import numpy as mat


def gaussian_kernel(xi, xj, sigma):
    gk = mat.subtract(xi, xj)
    gk = mat.dot(gk.transpose(), gk)

    gk = mat.divide(-gk, 2*(sigma * sigma))
    gk = mat.exp(gk)
    return gk
