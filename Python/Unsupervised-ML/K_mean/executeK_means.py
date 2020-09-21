# this file contains methods for performing K-mens clustering algorithm
from random import random

import numpy as mat
from matplotlib import pyplot
import os
from K_mean import nearestCentroid, calibrateCentroids, plotdata


def startK_means(X, initial_centroids, max_iter, visualize):
    centroids = initial_centroids
    previous_centroids = initial_centroids
    for i in range(max_iter):
        # find nearest centroids
        values, index = nearestCentroid.find(X, centroids)
        # calibrate centroids
        previous_centroids = centroids
        centroids = calibrateCentroids.calibrate(X, index, initial_centroids.shape[0])
        if visualize == 'yes':
            pyplot.title("iteration = "+str(i+1))
            plotdata.plot(X, index, centroids.shape[0], centroids, previous_centroids)
            pyplot.pause(0.05)
        print("k-means running",i+1,"/",max_iter,"complete")
        os.system("pause")

    pyplot.show()
