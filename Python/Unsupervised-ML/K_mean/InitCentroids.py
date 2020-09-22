# this file contain method which will initialize centroids randomly
from random import random

import numpy as mat


def getCentroids(X, numCentroids):
    centroids = mat.zeros((numCentroids, X.shape[1]))
    ind = mat.arange(X.shape[0])
    mat.random.shuffle(ind)
    centroids= X[ind[0:numCentroids],:]

    return centroids

