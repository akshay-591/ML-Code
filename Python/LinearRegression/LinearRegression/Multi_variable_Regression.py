# code for  multi variable Regression
# the data in this program is taken from the course assignment of Machine Learning by andrew ng on Coursera
# for learning purpose and all the python code is self written.
import time

import numpy
import cost_function
import Linear_hypothesis
import scipy.optimize as opt
import optimal_grad
import Feature_Normalisation

start = time.time()
# loading Data
print("Loading Data..............")
multivar_data = numpy.loadtxt(fname="ex1data2.txt", delimiter=",")

row = multivar_data.shape[0]
clm = multivar_data.shape[1]

x = multivar_data[0:row, 0:clm - 1]
y = multivar_data[0:row, clm - 1]
print("\nData loaded...........")
# feature Normalization

# initializing theta
theta = numpy.zeros(clm)
theta = numpy.c_[theta]

# calculate cost
cos = cost_function.cal_cost(x, y, theta)
print("initial cost = ")
print(cos)

# ===================================================== Normalizing features

normalize_feature = Feature_Normalisation.feature_normalisation(x)


# calculating optimal parameters by normal equation
row_size = x.shape[0]
x1 = numpy.c_[numpy.ones(row_size), x]
arr = numpy.dot(x1.transpose(), x1)
inverse = numpy.linalg.inv(arr)
step2 = numpy.dot(x1.transpose(), y)
norm_theta = numpy.dot(inverse, step2)

print("\n\n=====================optimum parameters using Normal equation============== ")
print(norm_theta)

# checking result
print("Estimate the price of a 1650 sq-ft, 3 br house")
x2 = numpy.array([[1650, 3]])
price = Linear_hypothesis.hypo(x2, norm_theta.transpose())
print(price)

# =======================================working=================================================================
initial_theta = numpy.zeros(clm)

Result = opt.minimize(fun=optimal_grad.cost,
                      x0=initial_theta,
                      method="Powell",
                      args=(x, y))

opt_theta = Result.x
print("\n\n===================optimum theta using Gradient Descent ===============")
print(opt_theta)

# checking result
print("Estimate the price of a 1650 sq-ft, 3 br house")
x2 = numpy.array([[1650, 3]])
price = Linear_hypothesis.hypo(x2, opt_theta)
print(price)
print("\n=========================================Program End==================================================")
# =====================================================Code END ===================================================
