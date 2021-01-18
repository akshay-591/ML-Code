"""
This File will execute the process of Learning.
"""
import numpy as mat
from scipy import optimize

from DeepNetsOptimization import WeightInit, Prediction
from DeepNetsOptimization.BackPropagation import BackProp
from DeepNetsOptimization.LossFunction import Loss


class Learn:
    def __init__(self, Input, Output, HiddenActivation=None, OutputActivation=None, AutoParameters=True, param=None,
                 InputLayerUnits=None, OutputLayerUnits=None, numHiddenLayers=None, HiddenLayerUnits=None, maxIter=None,
                 lamb=None):
        """
        This Method will starts the process of learning.

        :param Input: Input matrix.

        :param Output Output Matrix.

        :param AutoParameters: if True User dose not have to provide initial parameters By default it is true.

        :param param: if AutoParameters if false User have to provide the initial parameters.

        :param InputLayerUnits: Number of Units in InputLayer if User does not provide Total number of Columns will be
                                Considered as InputLayerUnits.

        :param OutputLayerUnits: Total Number of classes.

        :param numHiddenLayers: Number of Hidden Layer By default it is 1.

        :param HiddenLayerUnits: Number of Hidden Layer units. If Model have More than one one Hidden Layer user the
                                 list and add the number of neurons in that list respectively.
        :returns: Container.
        """
        if numHiddenLayers is None:
            numHiddenLayers = 1
        if InputLayerUnits is None:
            InputLayerUnits = Input.shape[1]  # Take the Column of Input as InputLayerUnits
        if lamb is None:
            lamb = 0
        if HiddenActivation is None:
            HiddenActivation = "ReLu"
        if OutputActivation is None:
            OutputActivation = "Sigmoid"
        if AutoParameters:
            self.initialParam = WeightInit.init(InputLayerUnits, OutputLayerUnits, numHiddenLayers, HiddenLayerUnits)
        else:
            self.initialParam = param

        self.HiddenActivation = HiddenActivation
        self.OutputActivation = OutputActivation
        self.X = Input
        self.Y = Output
        self.learnedWeights = mat.zeros(self.initialParam.shape)
        self.maxIter = maxIter
        self.InputLayerUnits = InputLayerUnits
        self.OutputLayerUnits = OutputLayerUnits
        self.numHiddenLayers = numHiddenLayers
        self.HiddenLayerUnits = HiddenLayerUnits
        self.prediction = mat.zeros(Output.shape)
        self.accuracy = 0
        self.lamb = lamb


def startTraining(Model):
    """
    This Method Starts the Training Process.

    :param Model: The Container or object of Learn Class in NNLearn
    :return: Model Containing Updated Result.
    """
    # Optimize
    result = optimize_grad(param=Model.initialParam,
                           maxiter=Model.maxIter,
                           args=(Model.X, Model.Y, Model.InputLayerUnits, Model.OutputLayerUnits, Model.numHiddenLayers,
                                 Model.HiddenLayerUnits, Model.HiddenActivation, Model.OutputActivation, Model.lamb))

    Model.learnedWeights = result.x
    # Compute Prediction and accuracy
    prediction, accuracy = Prediction.predict(Model, Model.X, Accuracy=True)
    Model.prediction = prediction
    Model.accuracy = accuracy

    return Model


def optimize_grad(param, maxiter=None, args=()):
    """
            This methods Uses built in Conjugate Gradient method for optimization.

            :param param: are the parameters which user want to optimize.
            :param maxiter: maximum iteration.
            :param args: rest of the parameters for ex- Regularization parameter or Output etc.
            :return: Optimized parameters.
            """
    if maxiter is None:
        result = optimize.minimize(fun=Loss,
                                   x0=param,
                                   args=args,
                                   method='CG',
                                   jac=BackProp)
    else:
        result = optimize.minimize(fun=Loss,
                                   x0=param,
                                   args=args,
                                   method='CG',
                                   jac=BackProp,
                                   options={'maxiter': maxiter})
    return result
