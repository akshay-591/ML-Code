"""
This file Contains the derivatives of Activation function.
"""
import numpy as mat


def derive(Input, Activation):

    global ds
    if Activation == "Sigmoid":
        ds = mat.multiply(Input, mat.subtract(1, Input))

    if Activation == "ReLu":
        ds = mat.where(Input <= 0, 0, 1)
    return ds
