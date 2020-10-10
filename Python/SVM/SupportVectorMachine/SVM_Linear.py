# This file contains code for SVM Model int this model since it is Linear classification problem

import numpy as mat
import scipy.io as io
from matplotlib import pyplot as plot
from SupportVectorMachine import GBO, SMO, OriginalSMO, Visualizeboundary
from scipy import optimize

# load data

data = io.loadmat('../Data/ex6data1.mat')

# extract input and output variable

X = data['X']
Y = data['y']
# Replacing 0 with -1 in Y dataset for ease of calculation
Y = mat.where(Y == 0, -1, Y)
# plotting data

positive_examples = X[mat.ix_(mat.where(Y == 1)[0])]
negative_examples = X[mat.ix_(mat.where(Y == -1)[0])]

plot.title("Support Vector Machine")
plot.xlabel("X1")
plot.ylabel("X2")
plot.plot(positive_examples[:, 0], positive_examples[:, 1], "+",
          negative_examples[:, 0], negative_examples[:, 1], "o")
plot.legend(["Positive", "Negative"])
plot.show()

# ======================================= Using SMO ======================================================
print("\nOptimizing using Sequential Minimal Optimization algo ================================== ")
X = data['X']
C = 1
# vectorised dataset
k = SMO.linear_kernel(X)
smo_model = SMO.simplifiedSMO(X, Y, C, k)
r = SMO.execute_SMO(smo_model, 20)

print("\n b = ", r.b)
print("\n W = ", r.W)

# ============================ Testing original SMO==================================
print("================ Testing Original SMO ======================")
model = OriginalSMO.SMO(X, Y, C, k)
new_model = OriginalSMO.execute(model)
print("done")
print("b =", new_model.b, "\nW = ", model.W)

# ================================================= Using Gradient Based Optimization =================================
print("\nOptimizing using GBO/ Soft margin algo============================================================")
X = mat.c_[mat.ones(X.shape[0]), X]
initial_theta = mat.zeros(X.shape[1])
C = 13
# calculate cost
cost = GBO.cost(initial_theta, X, Y, C)
Result = optimize.minimize(fun=GBO.cost,
                           x0=initial_theta,
                           args=(X, Y, C),
                           method="TNC",
                           jac=GBO.grad)

opt_grad = mat.c_[Result.x]
print("b = ", opt_grad[0])
print("W = ", opt_grad[1:opt_grad.shape[0], :])


# ================================================== plotting Decision boundary ========================================
Visualizeboundary.visualize_linear(new_model)
