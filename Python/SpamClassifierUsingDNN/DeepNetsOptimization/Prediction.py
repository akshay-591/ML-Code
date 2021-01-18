"""
This file contains Method which will compute the prediction in form of original labels and return its prediction
and model accuracy if user wants.
"""
import numpy as mat
from DeepNetsOptimization import ForwardPropagation, Reshape


def predict(model, X, Y = None, Accuracy = False):
    """
    This method calculates prediction and accuracy of the model.


    :param X: Input
    :param Y: Output if None will be considered what is provided in the 'model' Container.
    :param model: Container Containing all the Information. Like Number of Neurons in Layers and number of Hidden Layers
                  and Learned weights.
    :param Accuracy: if True will be calculated for the model.
    :return: Prediction and Accuracy if True.
    """

    global accuracy
    reshped_weights = Reshape.reshapeWeights(model.learnedWeights, model.InputLayerUnits,
                                             model.OutputLayerUnits, model.numHiddenLayers,
                                             model.HiddenLayerUnits)

    FFmodel = ForwardPropagation.Feed(X, reshped_weights, model.HiddenActivation,model.OutputActivation)

    Output_layer = FFmodel.ActivatedOutputs['output_layer']
    prediction = mat.where(Output_layer >= 0.5, 1, 0)


    if Accuracy:
        # calculate accuracy
        if Y is None:
            Y = model.Y
        prediction = mat.subtract(mat.c_[prediction], Y)
        accuracy = ((len(mat.where(prediction == 0)[0])) / len(Y)) * 100
        return prediction, accuracy

    return prediction
