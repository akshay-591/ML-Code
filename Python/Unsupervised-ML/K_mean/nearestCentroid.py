# this file contains methods which will find out the closest centroid to them example and will returns their index.

import numpy as mat


def find(X, centroids):
    temp = mat.zeros((X.shape[0], centroids.shape[0]))

    for i in range(centroids.shape[0]):
        distance = mat.subtract(X, centroids[i, :])
        square_distance = mat.power(distance, 2)
        square_distance = mat.sum(square_distance, axis=1)
        temp[:, i] = square_distance

    min_distance = mat.min(temp, axis=1)
    min_distance_index = mat.argmin(temp, axis=1)

    return min_distance, min_distance_index
