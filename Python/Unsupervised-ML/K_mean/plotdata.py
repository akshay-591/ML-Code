import numpy as mat
from matplotlib import pyplot


def plot(X, index, numCentroids,centroids,previous_centroids):
    for i in range(numCentroids):
        ind = mat.where(index == i)[0]
        r = mat.random.random()
        g = mat.random.random()
        b = mat.random.random()
        color = (r,g,b)
        pyplot.plot(previous_centroids[i,0],previous_centroids[i,1], color='black', linestyle='None', marker="+",
                    markersize=10, markeredgewidth=3)
        pyplot.plot(X[mat.ix_(ind)][:, 0], X[mat.ix_(ind)][:, 1],color=color,linestyle='None',marker= ".")
        pyplot.plot((previous_centroids[i, 0],centroids[i, 0]), (previous_centroids[i, 1],centroids[i, 1]),color='black')
