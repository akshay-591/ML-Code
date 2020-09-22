# This file contains the ML algorithm K_mean which is used for sorting out similar element/data together without
# knowing the output.

import numpy as mat
from scipy import io
from K_mean import nearestCentroid, calibrateCentroids,executeK_means,InitCentroids
from matplotlib import pyplot,image
# ======================== Testing model =======================
print("\n===================== Testing model =============== ")
# number of centroids
numCentroids = 3
initial_centroids = mat.array([[3, 3], [6, 2], [8, 5]])

# load data
data = io.loadmat('ex7data2.mat')
X = data['X']
distances, indexes = nearestCentroid.find(X, initial_centroids)
print("centroids for first three example should be = 0, 2, 1")
print("\nCentroids for the first 3 examples are")
print(indexes[0:3])

# ================== Calibrate Centroids ===============================
print("\n===== Testing Calibrate Centroids method ======= ")
new_Centroids = calibrateCentroids.calibrate(X, indexes, numCentroids)
print("values for new centroids should be = \n[[ 2.428301 3.157924 ]\n [ 5.813503 2.633656 ]\n [ 7.119387 3.616684 ]]\n")
print("new centroids values are")
print(new_Centroids)

# ==================== K-mean clustering ===================
print("Running K-mean Clustering ")
executeK_means.startK_means(X,initial_centroids, 10, False)

# ================== K-mean on Image pixels for Image Compression =============
print("\nrunning K-mean on Image pixels for Image Compression")
image_data = io.loadmat('bird_small.mat')
A = image_data['A']


A = mat.divide(A,255) # scale every value between 0-1
new_X = mat.reshape(A,((A.shape[0]*A.shape[1]),3)) # convert 3D array to 2D for ease of calculation

numCentroids = 16 # number of centroids
max_iter = 10
# select centroids
initial_Centroids = InitCentroids.getCentroids(new_X, numCentroids)
new_Centroids, indx = executeK_means.startK_means(new_X, initial_Centroids, max_iter, False)

values, idx = nearestCentroid.find(new_X, new_Centroids)
# recover compressed image
X_recovered = new_Centroids[idx, :]
X_recovered = mat.reshape(X_recovered,(A.shape[0], A.shape[1],3)) # reshape back to 3D
print(X_recovered.shape)
# plot both images side by side to see the difference

# plot original
fig = pyplot.figure()
ax=fig.add_subplot(1,2,1)
pyplot.title('original')
image_plot = pyplot.imshow(A)

# plot compressed
ax1 = fig.add_subplot(1, 2, 2)
pyplot.title('Compressed')
im_plot = pyplot.imshow(X_recovered)
pyplot.show()