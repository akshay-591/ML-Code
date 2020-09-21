# This file contains the ML algorithm K_mean which is used for sorting out similar element/data together without
# knowing the output.

import numpy as mat
from scipy import io
from K_mean import nearestCentroid, calibrateCentroids,executeK_means

# ======================== Testing model =======================
print("\n===================== Testing model =============== ")
# number of centroids
c = 3
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
new_Centroids = calibrateCentroids.calibrate(X,indexes,c)
print("values for new centroids should be = \n[[ 2.428301 3.157924 ]\n [ 5.813503 2.633656 ]\n [ 7.119387 3.616684 ]]\n")
print("new centroids values are")
print(new_Centroids)

# ==================== K-mean clustering ===================
executeK_means.startK_means(X,initial_centroids,10,'no')
