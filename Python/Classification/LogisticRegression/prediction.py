# this file has method to calculate accuracy of a model

import numpy as mat
from LogisticRegression import sigmoidFuntion


def checkAccuracy(optimised_Parameter, X, Y):
    # calculate prediction

    hypo = sigmoidFuntion.hypo(optimised_Parameter, X)
    prediction = sigmoidFuntion.sigmoid(hypo)
    prediction = mat.where(prediction >= 0.5, 1, prediction)
    prediction = mat.where(prediction < 0.5, 0, prediction)
    prediction = mat.subtract(mat.c_[prediction], Y)

    # calculate accuracy
    accuracy = ((len(mat.where(prediction == 0)[0]))/len(Y))*100
    return int(accuracy)
