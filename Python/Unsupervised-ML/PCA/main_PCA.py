"""
This File Contain PCA algorithm which is most used for Converting Higher dimension Data in Lower dimension without
Loosing much information. It can help in Visualization and Calculation both
"""
import numpy as mat
from scipy import io
from matplotlib import pyplot
from PCA import FindComponents, Projection, Decode
from MathsTools import FeatureNormalization

# load dataset
data = io.loadmat('../Data/ex7data1.mat')
X = data['X']

# plot data
print("Ploting example dataset")
pyplot.plot(X[:, 0], X[:, 1], '.')
pyplot.show()

# ====================== Principal Component Analysis =====================
print("running PCA on example dataset")
# Normalize dataset
X_norm, mu, sig = FeatureNormalization.normalize(X)
# calculate Coordinate matrix
# here U matrix contains Principal Component from best to worst and coordinate value of the data point on those PCs
U, S = FindComponents.execute(X_norm)
print("Values at first Principal component is = ", U[:, 0])

# plot Principal component on data
pyplot.plot(X[:, 0], X[:, 1], '.')
y = (mu + 1.5 * S[0] * U[:, 0])
y1 = (mu + 1.5 * S[1] * U[:, 1])
pyplot.plot((mu[0], y[0]), (mu[1], y[1]))
pyplot.plot((mu[0], y1[0]), (mu[1], y1[1]))
pyplot.show()

# ======================== Dimension Reduction =========================
# project data
numPCs = 1
X_project = Projection.project(X_norm, U, numPCs)
recover_X = Decode.recover(X_project, U, numPCs)
# plot data of both reduced and recovered space
pyplot.plot(X_norm[:, 0], X_norm[:, 1], '.')
pyplot.plot(recover_X[:, 0], recover_X[:, 1], "ro")

for i in range(X_norm.shape[0]):
    pyplot.plot((X_norm[i, 0], recover_X[i, 0]), (X_norm[i, 1], recover_X[i, 1]), linestyle="--")
pyplot.show()

# ============ Applying PCA on Face images data ==================
print("Applying PCA on Face images data")
# load face data set

face_data = io.loadmat('../Data/ex7faces.mat')
X_faces = face_data['X']
# plot 100 images
# plotting 100 images

fig, axes = pyplot.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    im = mat.reshape(X_faces[i, :], (32, 32))
    ax.imshow(im.transpose(), 'gray')
pyplot.show()

# =================== find out eigen faces ============
print("Running PCA on faces")
X_faces_norm, mu, sig = FeatureNormalization.normalize(X_faces)
U, S = FindComponents.execute(X_faces_norm)
print("plotting eigen faces")

fig, axes = pyplot.subplots(6, 6, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    im = mat.reshape(U[:, i], (32, 32))
    ax.imshow(im.transpose(), 'gray')
pyplot.show()

# ===================== Reducing eigen faces ========================
print("Reducing Dimension of faces")
numPCs = 100
z = Projection.project(X_faces_norm, U, numPCs)
print("new size of images is = ", z.shape)
recover_faces = Decode.recover(z,U,numPCs)

print("plotting Recovered images")

fig, axes = pyplot.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    im = mat.reshape(recover_faces[i,:], (32, 32))
    ax.imshow(im.transpose(), 'gray')
pyplot.show()

