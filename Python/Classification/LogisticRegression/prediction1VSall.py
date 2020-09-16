# this file contain method for one vs all prediction

import numpy as mat
from LogisticRegression import sigmoidFuntion


def predict(parameters, X, Y):
    prediction = sigmoidFuntion.sigmoid(sigmoidFuntion.hypo(parameters, X))
    prediction = mat.argmax(prediction, axis=1)

    prediction = mat.subtract(mat.c_[prediction], Y)

    # calculate accuracy
    accuracy = ((len(mat.where(prediction == 0)[0])) / len(Y)) * 100
    return int(accuracy)
