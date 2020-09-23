# This file contains Appliation of K-means and PCA

import numpy as mat
from matplotlib import pyplot
from scipy import io
from K_mean import InitCentroids, executeK_means
from PCA import Projection

# load image
from PCA import feature_normalization, RunPCA

image_data = io.loadmat('bird_small.mat')
A = image_data['A']
A = mat.divide(A, 255)  # scale every value between 0-1
new_X = mat.reshape(A, ((A.shape[0] * A.shape[1]), 3))  # convert 3D array to 2D for ease of calculation
numCentroids = 16  # number of centroids
max_iter = 10
# select centroids
initial_Centroids = InitCentroids.getCentroids(new_X, numCentroids)
new_Centroids, idx = executeK_means.startK_means(new_X, initial_Centroids, max_iter, False)

sel = mat.abs(mat.random.rand(1000) * new_X.shape[0] - 1)  # create array of 1000 numbers
sel = sel.astype(dtype=int)
temp_X = new_X[sel, :]  # select 1000 random indices
temp_idx = idx[sel]  # reduce size of idx array to 1000

# Visualizing in 3D
fig = pyplot.figure(figsize=(10, 10))
ax = pyplot.axes(projection='3d')
pyplot.title("Plotting 1000 random indices in 3D")
for i in range(numCentroids):
    ind = mat.where(temp_idx == i)[0]
    r = mat.random.random()
    g = mat.random.random()
    b = mat.random.random()
    color = (r, g, b)
    ax.scatter3D(temp_X[mat.ix_(ind)][:, 0], temp_X[mat.ix_(ind)][:, 1], temp_X[mat.ix_(ind)][:, 2], color=color)

pyplot.show()

# as this image pixel data is in high dimension we cannot visualize it on 2D graph so we are going to reduce its
# dimension using PCA and plot it on graph

# Normalize dataset
X_norm, mu, sig = feature_normalization.normalize(new_X)
# calculate Coordinate matrix
# here U matrix contains Principal Component from best to worst and coordinate value of the data point on those PCs
U, S = RunPCA.execute(X_norm)
Z = Projection.project(X_norm, U, 2)
temp_Z = Z[sel, :]

pyplot.title("Plotting in 2D after reducing dimension Using PCA")
for i in range(numCentroids):
    ind = mat.where(temp_idx == i)[0]
    r = mat.random.random()
    g = mat.random.random()
    b = mat.random.random()
    color = (r, g, b)
    pyplot.plot(temp_Z[mat.ix_(ind)][:, 0], temp_Z[mat.ix_(ind)][:, 1], color=color, linestyle='None', marker=".")

pyplot.show()
