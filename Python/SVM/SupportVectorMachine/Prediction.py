# this file contains the prediction method for SVM

import numpy as mat

from SupportVectorMachine import GaussianKernel


def predict(smo_Result, X, kernel):
    # it is efficient to only do calculation on those example whose relative lagrange multiplier is greater than 0
    # since rest of the alphas is going to be 0 so does the calculation.

    # find the index where lagrange multiplier is greater than 0
    global prediction
    index = mat.where(smo_Result.alphas > 0)[0]
    # clip the Y values from those indexes
    Y = smo_Result.Y[mat.ix_(index)]

    # clip the alphas value from those indexes
    alphas = smo_Result.alphas
    alphas = alphas[mat.ix_(index)]
    clip_X = smo_Result.X[index, :]
    if kernel == "linear":
        # if kernel is linear then calculate the prediction using X*W+b
        prediction = mat.dot(X, smo_Result.W) - smo_Result.b

    if kernel == "gaussian":
        # if kernel is RBF/gaussian first apply the kernel on every example than do the prediction
        X1 = mat.c_[mat.sum(mat.power(X, 2), 1)]
        X2 = mat.c_[mat.sum(mat.power(clip_X, 2), 1)].transpose()
        kernel_dataset = mat.dot(X, clip_X.transpose()) * -2 + X2 + X1
        kernel_dataset = mat.power(GaussianKernel.gaussian_kernel(1, 0, smo_Result.sigma), kernel_dataset)

        # do prediction
        kernel_dataset = mat.multiply(alphas.transpose(), mat.multiply(Y.transpose(), kernel_dataset))
        prediction = mat.sum(kernel_dataset, 1)

    # convert the result in binary form
    prediction = mat.where(prediction >= 0, 1, 0)
    return prediction
