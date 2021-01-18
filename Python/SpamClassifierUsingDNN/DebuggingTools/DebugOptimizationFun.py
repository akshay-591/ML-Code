"""
This file Contains Method which will debug the DeepNetsOptimization Functions Using Numerical Vs Analytical approach
"""

import numpy as mat
from DebuggingTools import WeightsDebug, TestNumericalGradient
from DeepNetsOptimization import LossFunction, BackPropagation


def debug(HiddenActivation, OutputActivation, InputUnits=None, OutputUnits=None, numHiddenLayer=None, HiddenLayerUnits=None, numExample=None,
          lamb=None):
    """
    This method will generate some random data and and Calculate Gradients or Derivative of Function Numerically and
    Analytical both.User either Can feed in there Input for Generating random data or Use the default values.This
    Methods are designed in a way that Input matrix have Examples in row and features in Columns.

    :param HiddenActivation:
    :param OutputActivation:
    :param Activation: The Activation Function For ex- ReLu or Sigmoid.
    :param InputUnits: Number of Units in Input Layer. By default it is 6.
    :param OutputUnits: Number of Classes or Number of Units in Output Layer.By default it is 5.
    :param numHiddenLayer: Number of Hidden Layers.By default it is 1.
    :param HiddenLayerUnits: Number of Hidden Layer Units if passing more than 1 Layer data then use the List[].By
                             Default it will be 5 for every layers.
    :param numExample: Number of examples By default it is 5
    :param lamb: Regularization parameter By default it will be 0.
    :return: Numerical Gradients , Analytical Gradients and difference/error.
    """

    # Check and define some parameters for Random Data
    if HiddenActivation is None:
        HiddenActivation = "ReLu"
    if OutputActivation is None:
        OutputActivation = "Sigmoid"
    if numHiddenLayer is None:
        numHiddenLayer = 1
    if InputUnits is None:
        InputUnits = 6
    if numExample is None:
        numExample = 5
    if OutputUnits is None:
        OutputUnits = 4
    if HiddenLayerUnits is None:
        HiddenLayerUnits = []
        for i in range(numHiddenLayer):  # this loop is to tackle the situation where numHidden layer is passed but
            # HiddenLayerUnits is None.
            HiddenLayerUnits.append(5)
    if lamb is None:
        lamb = 0

    # Generate some Weight parameters
    param = WeightsDebug.init(InputUnits, OutputUnits, numHiddenLayer, HiddenLayerUnits)

    # Generate some input and output data
    X = WeightsDebug.generate(numExample, InputUnits - 1)
    Y = 1 + mat.reshape(mat.arange(start=1, stop=numExample + 1, step=+1), (numExample, 1)) % OutputUnits

    # Calculates numerical gradients

    numerical_values = TestNumericalGradient.NumGrad(function=LossFunction.Loss,
                                                     theta=param,
                                                     parameters=(X, Y, InputUnits, OutputUnits, numHiddenLayer,
                                                                 HiddenLayerUnits, HiddenActivation,OutputActivation, lamb))

    # Calculates Analytical gradients
    Analytical_values = BackPropagation.BackProp(param, X, Y, InputUnits, OutputUnits, numHiddenLayer, HiddenLayerUnits,
                                                 HiddenActivation,OutputActivation, lamb)

    # calculate difference
    mat_a = mat.subtract(numerical_values, Analytical_values)
    mat_b = mat.add(numerical_values, Analytical_values)
    # calculate norm
    diff = mat.linalg.norm(mat_a) / mat.linalg.norm(mat_b)

    print("\nNumerical Calculated Gradients = \n", numerical_values)
    print("\nAnalytical Calculated Gradients = \n", Analytical_values)
    print("\ndifference = ", diff)
    print("\nif the both the Values are almost same and Difference  is less than 1e-9 than test is Successful.")

    return numerical_values, Analytical_values, diff
