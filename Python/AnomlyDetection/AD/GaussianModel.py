"""
This file contain method which will calculate multivariate Normal/Gaussian Distribution for the given dataset
Details provided on this link - http://cs229.stanford.edu/section/gaussians.pdf
"""

import numpy as mat
from AD import stand_dev


def calc_Gaussian(X,mn,sigma2):
    """""
    This method calculate the Probability Distribution using Gaussian Model for Multivariate Dataset
    """
    # calculate mean
    num_dimensions = X.shape[1]

    # as we need inverse of sigma2 matrix we need a square matrix
    # if sigma2 is a matrix it will be treated as Covariance matrix
    # if it is a vector than make it a  matrix

    if sigma2.ndim == 1:
        sigma2 = mat.diag(sigma2)
    X = mat.subtract(X, mn)
    Z = mat.sum(mat.multiply(mat.dot(X, mat.linalg.pinv(sigma2)), X), axis=1) * -0.5
    argument_e = mat.exp(Z)
    # finding determinant of sigma2
    determinant = mat.power(mat.linalg.det(sigma2), -0.5)
    # finding Probability Distribution
    Px = mat.dot(mat.dot(mat.power((2 * 3.14), (-num_dimensions / 2)), determinant), argument_e)

    return Px


# for calculating parameters for Gaussian Model
def gaussian_para(X):
    """
    for calculating parameters of Multivariate Gaussian/Normal Distribution Model
    """
    # calculate mean
    mn = mat.mean(X, axis=0)
    # calculate standard deviation
    sigma = stand_dev.stnd(X, dim=1)
    # calculate variance
    sigma2 = mat.power(sigma, 2)

    return mn, sigma2
