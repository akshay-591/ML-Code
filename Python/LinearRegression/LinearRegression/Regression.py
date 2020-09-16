# code for single variable regression
# the data in this program is taken from the course assignment of Machine Learning by andrew ng on Coursera
# for learning purpose and all the python code is self written.

import time

import numpy
import cost_function
import Gradient_Descent
import Linear_hypothesis

start = time.time()
# reading data from csv/txt file
data = numpy.loadtxt(fname="ex1data1.txt", delimiter=",")

# checking and storing the the size of the 'data' array
row = data.shape[0]
clm = data.shape[1]

# extracting data where x represent population of the city  and y represents the profits
x = data[0:row, 0:clm-1]
y = data[0:row, clm-1]
print(x.shape)
print(y.shape)

# plotting data
title = "TESTING"
x_lab = "Population of the city in 10,000s"
y_lab = "profits in $10,000s"
plotsymbol = "+"
# plot_data.plot(x, y, title, x_lab, y_lab, plotsymbol)

# initializing theta
theta = numpy.zeros(clm)
# converting into 2D array
theta = numpy.c_[theta]
print(theta.shape)
# calculate cost
cos = cost_function.cal_cost(x, y, theta)
print("cost = ")
print(cos)

# calculating gradient
alpha = 0.01
iteration = 1500
optimum_theta = Gradient_Descent.cal_grad(x, y, theta, alpha, iteration)
print("optimum parameters are = ")
print(optimum_theta)

# plotting line
pred_y = Linear_hypothesis.hypo(x, optimum_theta)
# plot_data.linoverdata_plot(x, y, pred_y, title, x_lab, y_lab, "+")
end =time.time()
print(end-start)

# ======================================Code
# END=========================================================================
