""""
This File Contains Method which will Reshape the weights which are contained in a single flat list.
"""
import numpy as mat


def reshapeWeights(param, inputUnits, outputUnits, numHiddenLayer=None, numHiddenUnit=None):
    """
    This is method is used to reshape the flat array of parameters for Neural Network.

    :param param: flat array of Parameters in. Array must be flattened in 'C' format.
    :param inputUnits: number of units in Input layer.
    :param outputUnits: Number of units in output layer.
    :param numHiddenLayer: Number of Hidden Layer.
    :param numHiddenUnit: Number of of units in the hidden layers. User should send a list Containing the number of
                          units respectively.
    :return: Dictionary Containing reshaped weights and key for the particular weights is "weights0" for 1st layer
             "weights1" for second layer and so on...
    """
    if isinstance(numHiddenUnit, list):  # checking if argument is list or not if not convert it into a list
        numHiddenUnits = numHiddenUnit
    else:
        numHiddenUnits = [numHiddenUnit]

    numTillHere = 0  # this variable is useful to track the number of element till where we have reshaped parameters
    # matrix
    weights = {}

    for i in range(numHiddenLayer + 1):
        if i == 0:  # reshaping weights for initial layer.

            previousLayerUnit = inputUnits
            nexLayerUnits = numHiddenUnits[i]
            numTillHere = ((previousLayerUnit + 1) * nexLayerUnits)

            weights.update({"weights" + str(i): mat.reshape(param[0:numTillHere],
                                                            (nexLayerUnits, previousLayerUnit + 1))})
        else:

            if numHiddenLayer > i:  # reshape weights for Hidden layer
                previousLayerUnit = numHiddenUnits[i - 1]
                nexLayerUnits = numHiddenUnits[i]
                weights.update({"weights" + str(i): mat.reshape(param[numTillHere:
                                                                      numTillHere + (nexLayerUnits * (
                                                                              previousLayerUnit + 1))],
                                                                (nexLayerUnits, previousLayerUnit + 1))})

                numTillHere = numTillHere + (nexLayerUnits * (previousLayerUnit + 1))  # update the variable
            if numHiddenLayer == i:  # when we reached at output layer.
                previousLayerUnit = numHiddenUnits[i - 1]
                nexLayerUnits = outputUnits

                weights.update({"weights" + str(i): mat.reshape(param[numTillHere:param.shape[0]],
                                                                (nexLayerUnits, previousLayerUnit + 1))})
                numTillHere = param.shape[0]
    return weights
