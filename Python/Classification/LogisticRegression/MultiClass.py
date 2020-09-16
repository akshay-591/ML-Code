# this model is a Multi class classification model which predicts the Hand writen digits using one vs all technique

import numpy as mat
import scipy.io as io
from matplotlib import pyplot as plt
import scipy.optimize as opt
from LogisticRegression import LossandGradient,prediction1VSall


# load data

data = io.loadmat("ex3data1.mat")

# extracting inputs and outputs

# X array contains the pixel intensity information of 20*20 grayscale image and had total of 5000 examples
X = data['X']


# Y array contains the output from  0 to 9 total of 5000
Y = data['y']
Y = mat.where(Y == 10, 0, Y)

# plotting 100 images
fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    im = mat.reshape(X[i * 50, 0:400], (20, 20))
    ax.imshow(im.transpose(), 'gray')

plt.show()

# applying one vs all techniques to get 'k' different sets of optimum parameters each for each label

# add column of ones to X
X = mat.c_[mat.ones(X.shape[0]), X]
k=10
optimum_parameterset = mat.empty((X.shape[1], k))
for i in range (k):
    initial_theta = mat.zeros(X.shape[1])
    new_output = mat.where(Y == i, 1, 0)
    Result = opt.minimize(fun=LossandGradient.regularised_cost,
                          x0=initial_theta,
                          args=(X, new_output, 1),
                          method="TNC",
                          jac=LossandGradient.regularised_grad)

    opt_grad = Result.x
    optimum_parameterset[:, i] = opt_grad

accuracy = prediction1VSall.predict(optimum_parameterset, X, Y)
print("dataset accuracy is = ",accuracy,"%")

