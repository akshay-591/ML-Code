# This file contains model for anomaly detection . We have data set of feature which represent some properties
# a product using Gaussian or Normal Distribution model we are going to separate out the products which have very low
# probability to be passed.

import numpy as mat
from scipy import io
from matplotlib import pyplot
from AD import GaussianModel, ComputThreshold

mat.set_printoptions(formatter={'float_kind': '{:f}'.format})
# load data
data = io.loadmat("ex8data1.mat")
X = data['X']
Xval = data['Xval']
yval = data['yval']

# Visualize
pyplot.xlim(0, 30)
pyplot.ylim(0, 30)
pyplot.xlabel('Latency (ms)')
pyplot.ylabel('Throughput (mb/s)')
pyplot.title("Network Server Data")
pyplot.plot(X[:, 0], X[:, 1], ".")
pyplot.show()

mu, sigma2 = GaussianModel.gaussian_para(X)
Px = GaussianModel.calc_Gaussian(X,mu,sigma2)

# ================================== Visualizing the anomalous and non anomalous data =======================
x = mat.linspace(0, 35, 71)
X1, Y1 = mat.meshgrid(x, x)

flat_X = mat.c_[X1.flatten(), Y1.flatten()]
zx = GaussianModel.calc_Gaussian(flat_X, mu, sigma2)
zx_resahpe = mat.reshape(zx, X1.shape)

pyplot.xlabel('Latency (ms)')
pyplot.ylabel('Throughput (mb/s)')
pyplot.title("High probability(non-anomalous) data is in the centre")
pyplot.plot(X[:, 0], X[:, 1], "+")
pyplot.contour(X1, Y1, zx_resahpe, levels=mat.power(10, mat.round(mat.linspace(-20, 0, 7))))
pyplot.show()

""""
Now we are going to select a threshold based on cross-validation set which will decide which data is Anomalous 
and which is not.
We are going to choose Threshold using Numerical Evaluation Technique.

"""

# calculate probability on validation set
pval = GaussianModel.calc_Gaussian(Xval, mu, sigma2)
threshold, F1 = ComputThreshold.compute(yval, mat.c_[pval])
print("F1 score is ", F1)
print("Threshold = ", threshold)

# ============ visualize outliers ================
outliers = mat.where(Px < threshold)[0]

pyplot.xlabel('Latency (ms)')
pyplot.ylabel('Throughput (mb/s)')
pyplot.title("point in small circles are outliers")
pyplot.plot(X[:, 0], X[:, 1], "+")
pyplot.plot(X[outliers, 0], X[outliers, 1], "o",color='red', fillstyle='none')
pyplot.contour(X1, Y1, zx_resahpe, levels=mat.power(10, mat.round(mat.linspace(-20, 0, 7))))
pyplot.show()

# ======================= Multidimensional Dataset =================================================================
""""
In this dataset it is hard to determine Anomaly because of High dimensional and also because only few feature will able
to distinguish between anomaly and non-anomaly data point.
"""

print("\nComputing on Multidimensional Data set")
# load data set
data2 = io.loadmat('ex8data2.mat')
X2 = data2['X']
X2val = data2['Xval']
y2val = data2['yval']

mu, sigma2 = GaussianModel.gaussian_para(X2)
Px = GaussianModel.calc_Gaussian(X2, mu, sigma2)

# calculate probability on validation set
p_val = GaussianModel.calc_Gaussian(X2val, mu, sigma2)
threshold, F1 = ComputThreshold.compute(y2val, mat.c_[pval])
print("F1 score is ", F1)
print("Threshold = ", threshold)
print("outliers found = ", len(mat.where(Px < threshold)[0]))
