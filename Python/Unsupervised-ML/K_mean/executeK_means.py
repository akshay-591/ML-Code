# this file contains methods for performing K-mens clustering algorithm
from random import random


from matplotlib import pyplot
from K_mean import plotdata, InitCentroids


def startK_means(X, initial_centroids, max_iter, visualize):
    """
    This Methods initialize the KMeans alg.
    :param X: Unlabelled Data.
    :param initial_centroids: initial Centroids.
    :param max_iter: max iteration.
    :param visualize: if user wants Real Time Visualization or not.
    :return: New Centroids and Indexes
    """
    global index
    centroids = initial_centroids
    previous_centroids = initial_centroids
    for i in range(max_iter):
        # find nearest centroids
        values, index = InitCentroids.findNearest(X, centroids)
        # calibrate centroids
        previous_centroids = centroids
        centroids = InitCentroids.calibrate(X, index, initial_centroids.shape[0])
        if visualize:
            pyplot.title("iteration = " + str(i + 1))
            plotdata.plot(X, index, centroids.shape[0], centroids, previous_centroids)
            pyplot.pause(1)
        print("k-means running", i + 1, "/", max_iter, "complete")

    pyplot.show()
    return centroids, index
