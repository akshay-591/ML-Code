# This is Model which suggest weather a Microchip  gets Quality passed or not

import numpy as mat
from matplotlib import pyplot as plot
from LogisticRegression import LossandGradient
from sklearn.preprocessing import PolynomialFeatures
from LogisticRegression import prediction

import scipy.optimize as opt

# load data

data = mat.loadtxt(fname="ex2data2.txt", delimiter=",")

rows = data.shape[0]
clm = data.shape[1]

X = data[0:rows, 0:clm - 1]
Y = mat.c_[data[0:rows, clm - 1]]

# plot data
positive_smaples = X[mat.ix_(mat.where(Y == 1)[0])]
negative_smaples = X[mat.ix_(mat.where(Y == 0)[0])]

plot.title("MicroChip Record")
plot.xlabel("Quality Test 1")
plot.ylabel("Quality Test 2")
plot.plot(positive_smaples[:, 0], positive_smaples[:, 1], "+",
          negative_smaples[:, 0], negative_smaples[:, 1], "1")
plot.legend(["Passed ", "Failed"])
plot.show()

# add ones to the X matrix and initial parameters

trans = PolynomialFeatures(degree=6)
X = trans.fit_transform(X)

print("new X shape", X.shape)

initial_theta = mat.zeros(X.shape[1])

# calculate cost

cost = LossandGradient.regularised_cost(initial_theta, X, Y, 10)

print("initial Cost = ", cost)

# Calculating optimum parameter using built in optimize function

test_theta = mat.ones(X.shape[1])

cost = LossandGradient.regularised_cost(test_theta, X, Y, 10)
print("\nCost at test theta and lambda =10 is \n ", cost)

grad = LossandGradient.regularised_grad(test_theta, X, Y, 10)
print("\nGradient (only first five) at test theta and lambda =10 is \n ", grad[0:5])


Result = opt.minimize(fun=LossandGradient.regularised_cost,
                      x0=initial_theta,
                      args=(X, Y, 1),
                      method="TNC",
                      jac=LossandGradient.regularised_grad)

opt_grad = Result.x

print("\n\nchecking accuracy................")

print("accuracy with regularised lambda 1 is \n", prediction.checkAccuracy(opt_grad, X, Y), "%")



