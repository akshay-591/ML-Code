# this file contains code for Tips and some tricks for calibrating Machine Learning

import numpy as mat
from matplotlib import pyplot
from scipy import io, optimize

from MLTricks import Regularised_Linear_Cost_Grad as Rg, leaning_curve

# load data
data = io.loadmat("ex5data1.mat")
# training set
X = data['X']
Y = data['y']

# cross- validation set
Xval = data['Xval']
Yval = data['yval']

# test set
Xtest = data['Xtest']
Ytest = data['ytest']

# plot data
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
pyplot.plot(X, Y, '+')
print("Plotting data...")
pyplot.show()

# =================================== calculate cost =====================
# add ones in X matrix
x = mat.c_[mat.ones(X.shape[0]), X]
initial_theta = mat.ones(x.shape[1])
lamb = 1
# =================================== calculate cost =====================
print("=================================== calculate cost =====================")
cost = Rg.cal_cost(initial_theta, x, Y, lamb)
print("Cost at theta [1 1] and lambda = 1 should be 303.993192")
print("calculated cost is = ", cost)

# ===================== ============== calculate initial gradient =========================
print("===================== ============== calculate initial gradient =========================")
grad = Rg.cal_grad(initial_theta, x, Y, lamb)
print("\ninitial grad should be [-15.303016, 598.250744 ]")
print("Calculated grad are  = ", grad.transpose())

# =============================== Train Linear Regression =============
print("=============================== Train Linear Regression =============")
lamb = 0
initial_theta = mat.zeros(x.shape[1])
result = optimize.minimize(fun=Rg.cal_cost,
                           x0=initial_theta,
                           args=(x, Y, lamb),
                           method='cg',
                           jac=Rg.cal_grad)
cost = Rg.cal_cost(result.x, x,Y,lamb)
print("iteration = ", result.jac.shape[0], "|", cost)

# =============================== plotting data and line
pred_y = Rg.hypo(x, mat.c_[result.x])
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
print("Plotting data...")
pyplot.plot(X, Y, '+', X, pred_y)
pyplot.legend(["Data", "Training Data"])
pyplot.show()

# as we can see that line is not fitting the data this condition is called underfitting or high bias
# as our data is non-linear our line or hyperplane should also be non-linear

# =================================== Learning curve test ==========================
# learning curve is one of the best technique to solve the problem or to find out what is the problem.
# now we are going to what learning curve of high bias looks like

# we are going to draw the variation between cost/error per iteration and apply this technique
# on training and validation set

error_train, error_val, iteration = leaning_curve.curve(X, Y, Xval, Yval, 0)
pyplot.xlabel('iteration')
pyplot.ylabel('error')
print("Plotting data...")
pyplot.plot(iteration, error_train, iteration, error_val)
pyplot.legend(["training data", "Validation Data"])
pyplot.show()




