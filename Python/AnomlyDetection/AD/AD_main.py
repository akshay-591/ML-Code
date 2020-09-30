# This file contains model for anomaly/ defect detection . We have data set of feature which represent some properties
# a product using Gaussian or Normal Distribution model we are going to separate out the products which have very low
# probability to be passed.

import numpy as mat
from scipy import io
from matplotlib import pyplot
from AD import GaussianModel

mat.set_printoptions(formatter={'float_kind': '{:f}'.format})
# load data
data = io.loadmat("ex8data1.mat")
X = data['X']
Xval = data['Xval']
yval = data['yval']

# Visualize
pyplot.xlim(0,30)
pyplot.ylim(0,30)
pyplot.xlabel('Latency (ms)')
pyplot.ylabel('Throughput (mb/s)')
pyplot.title("Network Server Data")
pyplot.plot(X[:,0], X[:,1],".")
pyplot.show()

Px = GaussianModel.calc_Gaussian(X)

# ================================== Visualizing the anomalous and non anomalous data =======================
x = mat.linspace(0, 35, 71)
X1, Y1 = mat.meshgrid(x, x)

flat_X = mat.c_[X1.flatten(), Y1.flatten()]
zx = GaussianModel.calc_Gaussian(flat_X)
zx_resahpe = mat.reshape(zx, X1.shape)

pyplot.xlabel('Latency (ms)')
pyplot.ylabel('Throughput (mb/s)')
pyplot.title("High probability(non-anomalous) data is inside the region")
pyplot.plot(X[:,0], X[:, 1], "+")
print(mat.power(10, mat.round(mat.linspace(-20, 0, 7))).shape)
pyplot.contour(X1, Y1, zx_resahpe, levels=mat.power(10, mat.round(mat.linspace(-20, 0, 7))))
pyplot.show()

