# this file contain method which will initialize centroids randomly
from random import random

import numpy as mat


def getCentroids(X, numCentroids):
    """
    This Method Returns Centroids by choosing them Randomly from The Given data.
    :param X: Unlabelled Data.
    :param numCentroids: Number of Centroids user wants.
    :return: randomly chosen Centroids.
    """
    centroids = mat.zeros((numCentroids, X.shape[1]))
    ind = mat.arange(X.shape[0])
    mat.random.shuffle(ind)
    centroids = X[ind[0:numCentroids],:]

    return centroids

def calibrate(X, indexes, numCentroids):
    """
    This Methods Calibrate the Centroids and find the new ones.
    :param X: Unlabelled Data
    :param indexes: indexes of Centroids which each Data example belongs to, They can be found using findNearest()
                    Method.
    :param numCentroids: Number of Centroids.
    :return: new Updated Centroids
    """
    centroids = mat.zeros((numCentroids, X.shape[1]))
    for i in range(numCentroids):
        ind = mat.where(indexes == i)[0]
        centroids[i, :] = mat.mean(X[mat.ix_(ind)], axis=0)
    return centroids

def findNearest(X, centroids):
    """
    This methods find the minimum distance and Indexes for each Data example w.r.t Centroids and
    :param X: Unlabeled Data Example
    :param centroids: Centroids/
    :return: Array of Minimum Distance and Array of centroids Index.
    """
    temp = mat.zeros((X.shape[0], centroids.shape[0]))

    for i in range(centroids.shape[0]):
        distance = mat.subtract(X, centroids[i, :])
        square_distance = mat.power(distance, 2)
        square_distance = mat.sum(square_distance, axis=1)
        temp[:, i] = square_distance
    min_distance = mat.min(temp, axis=1)
    min_distance_index = mat.argmin(temp, axis=1)
    return min_distance, min_distance_index
