"""
This file Contains Code for Linear Regression Model for single Variable.
The Data used in this Model is Taken for ML course BY prof. Andrew ng on Coursera.
"""

import numpy
import plot_data, Optimization


# reading data from csv/txt file
data = numpy.loadtxt(fname="../Data/ex1data1.txt", delimiter=",")

# checking and storing the the size of the 'data' array
row = data.shape[0]
clm = data.shape[1]

# extracting data where x represent population of the city  and y represents the profits
x = data[0:row, 0:clm - 1]
y = data[0:row, clm - 1]

# plotting data
title = "Regression"
x_lab = "Population of the city in 10,000s"
y_lab = "profits in $10,000s"
plotsymbol = "+"
plot_data.plot(x, y, title, x_lab, y_lab, plotsymbol)

# initializing theta
theta = numpy.zeros(clm)
# add ones in X matrix
x1 = numpy.c_[numpy.ones(x.shape[0]), x]

# calculate initial cost
cost = Optimization.cal_cost(theta, x1, y)
print("Initial cost is ", cost)

# optimize gradients
alpha = 0.01
iteration = 1500
optimum_theta = Optimization.cal_grad(x1, y, theta, alpha, iteration)
print("optimum parameters are = ", optimum_theta)

# plotting line
pred_y = Optimization.hypo(x1, optimum_theta)
plot_data.linoverdata_plot(x, y, pred_y, title, x_lab, y_lab, "+")

# ======================================Code END======================================================================
