# this file contains method for calibrating centroids

import numpy as mat


def calibrate(X, indexes, numCentroids):
    centroids = mat.zeros((numCentroids, X.shape[1]))
    for i in range(numCentroids):
        ind = mat.where(indexes == i)[0]
        centroids[i, :] = mat.mean(X[mat.ix_(ind)],axis=0)
    return centroids
