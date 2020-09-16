# This file contains full version of Sequential Minimal Optimization according to paper presented by John platt's on SMO
# https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/


import numpy as mat

from SupportVectorMachine import GaussianKernel


# Class for creating container
class SMO:
    def __init__(self, X, Y, C, kernel):
        self.X = X
        self.Y = Y
        self.C = C
        self.kernel = kernel
        self.alphas = mat.zeros((X.shape[0], 1))
        self.b = 0
        self.tol = 0
        self.errors = mat.zeros((X.shape[0], 1))
        self.W = mat.zeros((X.shape[1], 1))


# methods for Data vectorisation for Linear and RBF/gaussian
def linear_kernel(X):
    k = mat.dot(X, X.transpose())
    return k


def gaussian_kernel(X, sigma):
    X2 = mat.c_[mat.sum(mat.power(X, 2), 1)]
    gaussian_k = mat.dot(X, X.transpose()) * -2 + X2.transpose() + X2
    gaussian_k = mat.power(GaussianKernel.gaussian_kernel(1, 0, sigma), gaussian_k)
    return gaussian_k
<<<<<<< HEAD


=======


>>>>>>> parent of 9e8a657... Original SMO updated -- objective Function
# method for computing objective function
def objective_func(alphas, model, fun):
    if fun == 1:
        objective_fun = mat.sum(alphas) - (0.5 * (mat.sum(
            (alphas * alphas.transpose()) * (model.Y * model.Y.transpose()) * model.kernel)))
        return objective_fun


def execute(model):
    number_example = model.X.shape[0]

    # calculate initial errors by calculating SVM output - original output
    model.W = mat.dot(model.X.transpose(), mat.multiply(model.alphas, model.Y))
    model.errors = (mat.dot(model.X, model.W) - model.b) - model.Y

    # initialize tolerance 10^-3
    model.tol = pow(10, -3)

    examine_all = 1
    num_changed = 0
    print_dots = 0
    print("Calculating", end=" ")
    # initialize loop
    while num_changed > 0 or examine_all == 1:
        # this code is for printing dots on run screen so that user can see algo is running
        if print_dots > 100:
            print_dots = 0
            print("\n")
        print(".", end="")

        num_changed = 0
        if examine_all == 1:
            for i in range(number_example):  # loop over entire data set this is a Heuristic one
                value, model = examine_Example(i, model)
                num_changed += value

        else:
            mnc = mat.where((model.alphas != 0) & (model.alphas != model.C))[0]
            for i in mnc:  # loop examples which are related to non bound  alphas
                value, model = examine_Example(i, model)
                num_changed += value

        if examine_all == 1:
            examine_all = 0
        elif num_changed == 0:
            examine_all = 1
        print_dots = print_dots + 1

    # update Weight vector
    model.W = mat.dot(model.X.transpose(), mat.multiply(model.alphas, model.Y))
    return model


def examine_Example(j, model):
    global i0
    # number of Example
    m = len(model.Y)

    # get output of j
    y2 = model.Y[j]
    # store old alpha at j
    alpha_j_old = mat.asscalar(model.alphas[j])

    r2 = model.errors[j] * y2
    # check for the KKT conditions according to equation 12 in paper
    if (r2 < -model.tol and alpha_j_old < model.C) or (r2 > model.tol and alpha_j_old > 0):
        # get the index of non bound alphas
        non_zero_non_c = mat.where((model.alphas != model.C) & (model.alphas != 0))[0]

        if len(non_zero_non_c) > 1:  # if number of non bound alphas is grater than 1 choose i1 acc.
                                     # to second heuristic on page no.9

            if model.errors[j] > 0:  # if current error is greater than 0 choose i where error minimum
                i0 = mat.asscalar(mat.argmin(model.errors))

            if model.errors[j] <= 0:  # if current error is less than or equal to 0 choose i where error is max
                i0 = mat.argmax(model.errors)

            result, model = take_step(i0, j, model)  # call method
            if result == 1:  # if result is true return true
                return 1, model

                # if above condition does not satisfy then loop over non bound alphas starting at random point
        for i in mat.roll(non_zero_non_c, mat.random.choice(mat.arange(m))):
            result, model = take_step(i, j, model)
            if result == 1:
                return 1, model
        # if above condition also not satisfied then loop over entire example set
        for i in mat.roll(mat.arange(m), mat.random.choice(mat.arange(m))):

            result, model = take_step(i, j, model)
            if result == 1:
                return 1, model

    return 0, model


def take_step(i1, i2, model):
    m = len(model.Y)
    if i1 == i2:  # if both index are same then return false and take different example
        return 0, model

    # store old alphas
    alpha_i_old = mat.asscalar(model.alphas[i1])
    alpha_j_old = mat.asscalar(model.alphas[i2])

    # store output at i1 and i2
    y1 = model.Y[i1]
    y2 = model.Y[i2]

    s = y1 * y2

    # compute L and H
    if y1 == y2:  # if both output are same compute by equation 14
        L = max(0, (model.alphas[i1] + model.alphas[i2] - model.C))
        H = min(model.C, (model.alphas[i1] + model.alphas[i2]))

    else:  # if both output are not same compute by equation 13
        L = max(0, (model.alphas[i2] - model.alphas[i1]))
        H = min(model.C, (model.C + model.alphas[i2] - model.alphas[i1]))

    if L == H:  # if L and H is equal then return false
        return 0, model

    # compute eta by equation 15
    eta = model.kernel[i1, i1] + model.kernel[i2, i2] - (2 * model.kernel[i1, i2])

    if eta > 0:  # if eta is negative compute new alpha at i2 using equation 16
        a2 = alpha_j_old + ((y2 * (model.errors[i1] - model.errors[i2])) / eta)
        # choose new alpha at i2 by equation 17
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H

    else:  # if eta is not negative then compute objective function
        alphas_temp = model.alphas.copy()
        alphas_temp[i2] = L
        L_obj = objective_func(alphas_temp, model, fun=1)  # compute at a2 = L
        alphas_temp[i2] = H
        H_obj = objective_func(alphas_temp, model, fun=1)  # compute at a2 = H

        # set alpha at i2 according to bound condition withing some tolerance
        if L_obj < (H_obj - model.tol):
            a2 = H
        elif L_obj > (H_obj + model.tol):
            a2 = L
        else:
            a2 = alpha_j_old

    if a2 < pow(10, -8):
        a2 = 0.0
    elif a2 > (model.C - pow(10, -8)):
        a2 = model.C

    if abs(a2 - alpha_j_old) < (model.tol * (a2 + alpha_j_old + model.tol)):
        return 0, model

    # compute alpha at i1 by equation  18
    a1 = alpha_i_old + s * (alpha_j_old - a2)

    # calculate b1 and b2 by equation 20 and 21
    b1 = model.b + model.errors[i1] + (y1 * (a1 - alpha_i_old) * model.kernel[i1, i1]) + (
            y2 * (a2 - alpha_j_old) * model.kernel[i1, i2])
    b2 = model.b + model.errors[i2] + (y1 * (a1 - alpha_i_old) * model.kernel[i1, i2]) + (
            y2 * (a2 - alpha_j_old) * model.kernel[i2, i2])

    # select threshold b according to condition
    if 0 < a1 < model.C:
        b_new = b1
    elif 0 < a2 < model.C:
        b_new = b2

    # when lagrange multipliers at bound that means all threshold between b1 and b2 satisfy KKT condition choose
    # an average value
    else:
        b_new = (b1 + b2) / 2

    # update new alphas
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # update the error cache
    # update the error at which index alphas are optimize to 0.0
    for index, al in zip([i1, i2], [a1, a2]):
        if 0.0 < al < model.C:
            model.errors[index] = 0.0

    # update non-optimized alphas errors
    non_optimize = [n for n in range(m) if (n != i1 and n != i2)]
<<<<<<< HEAD
    model.errors[non_optimize] = model.errors[non_optimize] \
                                 + y1 * (a1 - alpha_i_old) * mat.c_[model.kernel[i1, non_optimize]] + y2 * \
                                 (a2 - alpha_j_old) * mat.c_[model.kernel[i2, non_optimize]] + model.b - b_new

=======

    model.errors[non_optimize] = model.errors[non_optimize] \
                                 + y1 * (a1 - alpha_i_old) * mat.c_[model.kernel[i1, non_optimize]] + y2 * \
                                 (a2 - alpha_j_old) * mat.c_[model.kernel[i2, non_optimize]] + model.b - b_new
>>>>>>> parent of 9e8a657... Original SMO updated -- objective Function
    # update threshold
    model.b = b_new
    return 1, model
