"""
Binary Classification Model
"""

import numpy as mat
from matplotlib import pyplot as plot
from LogisticRegression import LossandGradient
from LogisticRegression import sigmoidFuntion
import scipy.optimize as opt


# load data
data = mat.loadtxt(fname="../Data/ex2data1.txt", delimiter=",")

rows = data.shape[0]
clm = data.shape[1]

X = data[0:rows, 0:clm-1]
Y = mat.c_[data[0:rows, clm-1]]

# plot data
positive_samples = X[mat.ix_(mat.where(Y == 1)[0])]
negative_samples = X[mat.ix_(mat.where(Y == 0)[0])]


plot.title("Students record")
plot.xlabel("Exam1 Score")
plot.ylabel("Exam2 Score")
plot.plot(positive_samples[:, 0], positive_samples[:, 1], "+",
          negative_samples[:, 0], negative_samples[:, 1], "1")
plot.legend(["Pass Students ", "Fail Students"])
plot.show()

# add ones to the X matrix and initial parameters
X = mat.c_[mat.ones(rows), X]
initial_theta = mat.zeros(clm)
lamb = 0
# calculate cost
cost = LossandGradient.regularised_cost(initial_theta, X, Y, lamb)
print("initial Cost = ", cost)

# Calculating optimum parameter using built in optimize function


Result = opt.minimize(fun=LossandGradient.regularised_cost,
                      x0=initial_theta,
                      args=(X, Y, lamb),
                      method="TNC",
                      jac=LossandGradient.regularised_grad)

opt_grad = Result.x
print("optimum grads are = ", opt_grad)

# calculating cost again

cost = LossandGradient.regularised_cost(opt_grad, X, Y, lamb)

print("final cost = ", cost)

print("Chance of student with Exam 1 score = 45 and Exam two score with = 85 getting admit or not  ",
      (sigmoidFuntion.sigmoid(mat.dot(mat.c_[1, 45, 85], opt_grad)))*100, "%")





