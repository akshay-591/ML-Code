"""
This File Contains Method for Generating some Data which is useful for Debugging Model.
"""

import numpy as mat


def init(InputUnits, OutputUnits, numHiddenLayer, HiddenUnits=None):
    """
    This methods is used to generate all parameters for Multiple layers at once.

    :param InputUnits: Number of Input Units.
    :param OutputUnits: Number of Output Units
    :param numHiddenLayer: Number of Hidden Layers
    :param HiddenUnits: Number of Units in Hidden Layers
    :return: flat array of Containing all parameters.
    """
    global HiddenUnit
    all_weights = []
    if HiddenUnits is None:
        HiddenUnits = []
    elif isinstance(HiddenUnits, int):
        HiddenUnits = [HiddenUnits]

    # for InputLayer

    parameters = generate(HiddenUnits[0], InputUnits)
    allWeights = mat.r_[parameters.flatten()]

    if numHiddenLayer > 1:
        for i in range(numHiddenLayer):
            if i < numHiddenLayer-1:
                parameters = generate(HiddenUnits[i+1], HiddenUnits[i])
                allWeights = mat.r_[allWeights, parameters.flatten()]
            else:
                parameters = generate(OutputUnits, HiddenUnits[i])
                allWeights = mat.r_[allWeights, parameters.flatten()]

    else:
        # for output layer
        parameters = generate( OutputUnits, HiddenUnits[0])
        allWeights = mat.r_[allWeights, parameters.flatten()]

    return allWeights


def generate(nextLayerUnits, FirstLayerUnits):
    """
    This method will Generate Data for Debugging.

    :param nextLayerUnits:
    :param FirstLayerUnits:
    :return: parameters of dimension nextLayerUnits*FirstLayerUnits+1
    """
    array = mat.arange(start=1, stop=(nextLayerUnits * (FirstLayerUnits + 1)) + 1, step=1)
    debugWeights = mat.divide(mat.reshape(mat.sin(array), (nextLayerUnits, (FirstLayerUnits + 1))), 10)

    return debugWeights
