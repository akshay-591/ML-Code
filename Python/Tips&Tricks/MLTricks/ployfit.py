# PLOTFIT Plots a learned polynomial regression fit over an existing figure
import numpy as mat
from MLTricks import Poly_mapping


def fit(min_x, max_x, mu, sig, p):
    # increasing the range of plot
    x = mat.c_[mat.arange(start=(min_x-25), stop=(max_x+15), step=0.05)]
    x_poly = Poly_mapping.map(x, p)
    x_poly = mat.subtract(x_poly, mu)
    x_poly = mat.divide(x_poly, sig)
    x_poly = mat.c_[mat.ones(x_poly.shape[0]), x_poly]

    return x, x_poly
