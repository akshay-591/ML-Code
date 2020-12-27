"""
This file Contains code for multiVariable Linear Regression where the number of features is more than one.
The Data used in this Model is Taken for ML course BY prof. Andrew ng on Coursera.
"""

import numpy
import Optimization
from MathTools import FeatureNormalization

import scipy.optimize as opt

# loading Data
print("Loading Data..............")
multivar_data = numpy.loadtxt(fname="../Data/ex1data2.txt", delimiter=",")

row = multivar_data.shape[0]
clm = multivar_data.shape[1]

x = multivar_data[0:row, 0:clm - 1]
y = multivar_data[0:row, clm - 1]
print("\nData loaded...........")

# feature Normalization
x_norm, mu, sig = FeatureNormalization.normalize(x)

# initializing theta
initial_theta = numpy.zeros(clm)
x1 = numpy.c_[numpy.ones(row), x_norm]

# calculate cost
cost = Optimization.cal_cost(initial_theta, x1, y)
print("initial cost = ", cost)

# ================================  Calculating optimal parameters by normal equation =============================

arr = numpy.dot(x1.transpose(), x1)
inverse = numpy.linalg.inv(arr)
step2 = numpy.dot(x1.transpose(), y)
norm_theta = numpy.dot(inverse, step2)

print("\n\n=====================optimum parameters using Normal equation============== ")
print(norm_theta)

# checking result
x2 = numpy.array([[1650, 3]])

# normalize
x2_norm = FeatureNormalization.normalize2(x2, mu, sig)
# add ones
x2_norm = numpy.c_[numpy.ones(x2.shape[0]), x2_norm]

price = Optimization.hypo(x2_norm, numpy.c_[norm_theta.transpose()])
print("Estimate the price of a 1650 sq-ft, 3 br house is  ", numpy.round(price.item(), 0))

# ======================================= Optimization using Gradient Descent =====================================

Result = opt.minimize(fun=Optimization.cal_cost,
                      x0=initial_theta,
                      method="CG",
                      args=(x1, y),
                      jac=Optimization.grad_opt)

opt_theta = Result.x
print("\n\n===================optimum theta using Gradient Descent ===============")
print(opt_theta)

# checking result
price = Optimization.hypo(x2_norm, opt_theta)

print("Estimate the price of a 1650 sq-ft, 3 br house is  ", numpy.round(price.item(), 0))

print("\n=========================================Program End==================================================")

# =====================================================Code END ===================================================
