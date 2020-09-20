# this file contains code for Tips and some tricks for calibrating Machine Learning

import numpy as mat
from matplotlib import pyplot
from scipy import io, optimize

from MLTricks import Regularised_Linear_Cost_Grad as Rg, leaning_curve, Poly_mapping, feature_normalization, ployfit

mat.set_printoptions(formatter={'float_kind':'{:f}'.format})
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
cost = Rg.cal_cost(result.x, x, Y, lamb)
print("iteration = ", result.jac.shape[0], "|", cost)

# =============================== plotting data and line =============================
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

error_train, error_val, iteration = leaning_curve.curve(mat.c_[mat.ones(X.shape[0]), X], Y, mat.c_[mat.ones(Xval.shape[0]), Xval], Yval, 0)
print(error_train)
pyplot.xlabel('iteration')
pyplot.ylabel('error')
print("Plotting data...")
pyplot.plot(iteration, error_train, iteration, error_val)
pyplot.legend(["training data", "Validation Data"])
pyplot.show()

# we can see in the graph the error is increasing with the number of training example and
# also error is high in cross validation set this is because of non linearity of the data

# To solve this problem we can add some more features to overcome the under fitting
# we can create new features using polynomial regression.

# we are going to take this existing feature and going to map them in it higher power
# our new feature will look like this --  x1 = (water level), x2 = (water level)^2, xp = (water level)^P

# mapping and normalizing training set
power = 8
X_mapped = Poly_mapping.map(X, power)
# print(X_mapped)
# normalize or scaling the mapped input matrix
X_mapped, mu, sig = feature_normalization.normalize(X_mapped)

# adding ones
X_mapped = mat.c_[mat.ones(X_mapped.shape[0]), X_mapped]
# mapping and normalizing test set
X_poly_test = Poly_mapping.map(Xtest, power)
X_poly_test = mat.subtract(X_poly_test, mu)
X_poly_test = mat.divide(X_poly_test, sig)
X_poly_test = mat.c_[mat.ones(X_poly_test.shape[0]), X_poly_test]  # add ones column

# mapping and normalizing cross validation set
X_poly_val = Poly_mapping.map(Xval, power)
X_poly_val = mat.subtract(X_poly_val, mu)
X_poly_val = mat.divide(X_poly_val, sig)
X_poly_val = mat.c_[mat.ones(X_poly_val.shape[0]), X_poly_val]  # add ones column

print("\n========================== normalized training example are ================")
print(X_mapped[0, :])

print(" ================  Training newly added Polynomial Regression ================ ")
lamb = 0
initial_theta = mat.zeros(X_mapped.shape[1])
result = optimize.minimize(fun=Rg.cal_cost,
                           x0=initial_theta,
                           args=(X_mapped, Y, lamb),
                           method='CG',
                           jac=Rg.cal_grad,
                           options={'maxiter': 200})

min_x = mat.min(X, axis=0)
max_x = mat.max(X, axis=0)
x, x_poly = ployfit.fit(min_x, max_x, mu, sig, power)

pred_y = Rg.hypo(x_poly, result.x)
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
print("Plotting data...")
pyplot.plot(X, Y, '+',x , pred_y)
pyplot.legend(["Data", "Training Data"])
pyplot.show()

# ================================== Learning Curve for newly added Polynomial Regression ===========
print(" ================  Learning Curve for newly added Polynomial Regression ================ ")
error_train1, error_val1, iteration = leaning_curve.curve(X_mapped, Y, X_poly_val, Yval, 1)
print(error_train1)
pyplot.xlabel('iteration')
pyplot.ylabel('error')
print("Plotting data...")
pyplot.plot(iteration, error_train1, iteration, error_val1)
pyplot.xlim(0, 13)
pyplot.ylim(0, 100)
pyplot.legend(["training data", "Validation Data"])
pyplot.show()

