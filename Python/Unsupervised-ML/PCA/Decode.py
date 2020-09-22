# this file contains method which will project back the data from subspace to Higher space

import numpy as mat


def recover(Z, U, numPcs):
    recoverd_X = mat.dot(Z, U[:, 0:numPcs].transpose())

    return recoverd_X
