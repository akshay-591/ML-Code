import numpy as mat
import scipy.io as io
from matplotlib import pyplot as plot
from SupportVectorMachine import GaussianKernel, SMO, Prediction, OriginalSMO, Visualizeboundary

# load data

data = io.loadmat('../Data/ex6data2.mat')

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

# =============================== Testing Kernel ======================================================================
print("\n=================Testing Gaussian Kernel=========================")
xi = mat.array([[1, 2, 1]])
xj = mat.array([[0, 4, -1]])
sigma = 2
ans = GaussianKernel.gaussian_kernel(xi.transpose(), xj.transpose(), sigma)
print("\nExpected value at sigma 2  should be  = 0.324652")
print("the value is = ", ans)

# ========================================= Optimizing using SMO =======================================================
print("\nrunning Simplified SMO for optimization===========================")
C = 2
sigma = 0.1
max_passes = 5
k = SMO.gaussian_kernel(X,sigma)
smo_model = SMO.simplifiedSMO(X, Y, C, k)
r = SMO.execute_SMO(smo_model, max_passes)

# ============================ Testing original SMO==================================
print("================ Testing Original SMO ======================")
model = OriginalSMO.SMO(X, Y, C, k)
model.sigma = sigma
new_model = OriginalSMO.execute(model)
print("done")
print(new_model.b, model.W)

# ========================================= Visualising boundary =======================================================

Visualizeboundary.visualize_nonLinear(new_model)
