"""
This file Contains feedForward and BackPropagation method. Which is just like Cost/error and Gradient Descent but in
Neural Network they are called by different name because of their structure and working.
"""
import numpy as mat
from DeepNetsOptimization.ForwardPropagation import Feed
from DeepNetsOptimization.Reshape import reshapeWeights


def Loss(param, X, Y, inputUnits, outputUnits, numHiddenLayer, numHiddenUnits, HiddenActivation, OutputActivation,
         lamb):

    """
    This method will calculate the cost/error Using Maximum Likelihood also known as Cross-entropy.

    :param param: flat array containing all weights (all layer).
    :param X: Input matrix.
    :param Y: output matrix.

    :param inputUnits: Input or initial layer units/Neurons mostly equal to total element of an image or number of
                       features.
    :param outputUnits: Output layer Units/Neuron which is equal to the number of classes.
    :param numHiddenLayer: Number of Hidden layers
    :param numHiddenUnits: Number of of units in the hidden layers. User should send a list Containing the number of
                           units respectively.
    :param HiddenActivation: Hidden Activation is the Activation Type Which user wants to perform in the
                                     Hidden Layers ex - ReLu or Sigmoid.
    :param OutputActivation: OutputActivation is the Activation Type Which user wants to perform in the Output
                                     Layer. ex- Sigmoid and ReLu.
    :param lamb: Regularization parameters.
    :return: Cost/error .
    """

    total_example = X.shape[0]

    weights = reshapeWeights(param, inputUnits, outputUnits, numHiddenLayer, numHiddenUnits)

    # call FeedForward Propagation and Calculate the outputs
    FFModel = Feed(X, weights, HiddenActivation, OutputActivation)

    # Now Calculate the error between Original Output and the prediction

    """now we have classes to the equal number of outputs what we are going to do now will perform 1 vs all 
    Technique. In this Technique we are going to take 1 set and calculate the error by comparing it with one output 
    from one Neuron/unit for ex -  if we have 10 classes than the output will going have 10 Columns ( depends on how 
    data is arranged) Now we will take 1st set in which label 1 is going to be equal to 1 and rest will be 0 and then 
    we are going to subtract this set with Column 1 (or we can say output from neuron/unit 1 of output layer) only 
    from output and similarly set 2 with Column 2 set 3 with Column 3 and so on and at the end will add all the 
    errors that will give us the total error """

    # Here we are going to maximum likelihood Technique to calculate the errors

    loss0 = mat.zeros((outputUnits, 1))
    loss1 = mat.zeros((outputUnits, 1))
    output = FFModel.ActivatedOutputs['output_layer']

    for i in range(outputUnits):
        new_output = mat.where(Y == i + 1, 1, 0)  # where Y == i replace with 1 and rest of the value to 0

        # Loss when when Y=1
        loss1[i, :] = mat.dot(-new_output.transpose(), mat.log(mat.c_[output[:, i]]))

        # loss when Y=0
        loss0[i, :] = mat.dot(mat.subtract(1, new_output).transpose(), mat.log(mat.subtract(1, mat.c_[output[:, i]])))

    # Total Avg loss

    loss_final = mat.multiply((1 / total_example), mat.sum(mat.subtract(loss1, loss0)))

    # Regularization regularize parameter = 1/2m * sum(theta(i)^2) from i=1 (i=0 or bias not included) to n where n
    # is number of features
    regularized_value = 0
    for i in range(len(weights)):
        regularized_Layer = mat.multiply(lamb / (2 * total_example),
                                         mat.sum(mat.power(
                                             weights['weights' + str(i)][:, 1:weights['weights' + str(i)].shape[1]],
                                             2)))
        regularized_value = regularized_value + regularized_Layer

    J = loss_final + regularized_value
    return J
