# this file contains the code for how to choose best possible value set of parameter C and sigma when
# working with RBF/ gaussian kernel using training set and cross validation set

import numpy as mat
import scipy.io as io
from matplotlib import pyplot as plot
from SupportVectorMachine import SMO, Prediction, OriginalSMO, Visualizeboundary

# load data

data = io.loadmat('ex6data3.mat')


# extract input and output variable

# training set
X = data['X']
Y = data['y']

# cross validation set
Xval = data['Xval']
Yval = data['yval']

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

# ======================================= Choosing sigma and C ====================================================
# we will going to try try out every combination of given values and whichever set will perform best on the
# cross validation set will going to be used for the prediction


sig_set = mat.c_[[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]]
C_set = mat.c_[[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]]
error_set = mat.zeros((len(sig_set), len(C_set)))

for i in range(C_set.shape[0]):
    for j in range(sig_set.shape[0]):
        C = C_set[i].item()
        sigma = sig_set[j].item()
        k = SMO.gaussian_kernel(X, sigma)
        model = OriginalSMO.SMO(X, Y, C, k)
        model.sigma = sigma
        new_model = OriginalSMO.execute(model)
        prediction = mat.c_[Prediction.predict(new_model, X=Xval, kernel="gaussian")]

        mean_error = mat.mean(abs(prediction-Yval))
        error_set[j, i] = mean_error

    print("============================ set ", i+1, " Complete===================")

# get the index where error is minimum
# here every has column is equal to the length of C_set and rows are equal to length of sig_set

C_index = mat.where(mat.min(error_set, axis=0) == mat.min(mat.min(error_set, axis=0)))[0]
sig_index = mat.where(mat.min(error_set, axis=1) == mat.min(mat.min(error_set, axis=1)))[0]

# ========================================= Visualising boundary =======================================================

C = C_set[C_index]
sigma = sig_set[sig_index]
k = SMO.gaussian_kernel(X, sigma)
model = OriginalSMO.SMO(X, Y, C, k)
model.sigma = sigma
new_model = OriginalSMO.execute(model)

Visualizeboundary.visualize_nonLinear(new_model)