# PlotFit Contains method which Calculate the data for non-Linear Line like For non-Linear learned polynomial regression
import numpy as mat
from MLTricks import Poly_mapping


def fit(min_x, max_x, mu, sig, p):
    # increasing the range of plot
    x = mat.c_[mat.arange(start=(min_x - 15), stop=(max_x + 25), step=0.05)]

    x_poly = Poly_mapping.mapPoly(x, p)  # map the polynomial feature

    # normalize
    x_poly = mat.subtract(x_poly, mu)  #
    x_poly = mat.divide(x_poly, sig)
    # add ones
    x_poly = mat.c_[mat.ones(x_poly.shape[0]), x_poly]

    return x, x_poly
