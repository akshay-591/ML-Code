# this file contain sigmoid function

import numpy as mat


def predict(x, theta, Activation=None):
    """
    This method calculate prediction for the Model.

    :param Activation: Activation Function ReLu and Sigmoid.
    :param x: input parameters.
    :param theta: weight parameters.
    :return: prediction.
    """

    # calculating prediction
    global ActivationPrediction
    if theta.shape[0] != x.shape[1] and theta.shape[1] == x.shape[1]:
        prediction = mat.dot(x, theta.transpose())
    else:
        prediction = mat.dot(x, theta)
    if Activation is None:
        print("None")
        return prediction
    if Activation == "Sigmoid":
        ActivationPrediction = mat.divide(1, mat.add(1, mat.exp(-prediction)))
    if Activation == "ReLu":
        ActivationPrediction = mat.where(prediction <=0, 0, prediction)
    return prediction, ActivationPrediction
