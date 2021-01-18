"""
This File Contain method which will compute the derivative using BackPropagation.
"""
import numpy as mat

from DeepNetsOptimization import ActivationDeriavtive
from DeepNetsOptimization.ForwardPropagation import Feed
from DeepNetsOptimization.Reshape import reshapeWeights


def BackProp(param, X, Y, inputUnits, outputUnits, numHiddenLayer, numHiddenUnits, HiddenActivation, OutputActivation,
             lamb):
    """
    This Method will execute BackPropagation algorithm and find the derivative w.r.t every parameters.

    :param param: flat array containing all weights (all layer).
    :param X: Input matrix.
    :param Y: output matrix.
    :param inputUnits: Input or initial layer units/Neurons mostly equal to total element of an image or number of
                       features.
    :param numHiddenLayer: Number of Hidden Layer.
    :param numHiddenUnits: Hidden layer Units/Neurons.
    :param outputUnits: Output layer Units/Neuron which is equal to the number of classes.
    :param HiddenActivation: Hidden Activation is the Activation Type Which user wants to perform in the
                                     Hidden Layers ex - ReLu or Sigmoid.
    :param OutputActivation: OutputActivation is the Activation Type Which user wants to perform in the Output
                                     Layer. ex- Sigmoid and ReLu.
    :param lamb: Regularization parameters.
    :return: derived function values with respect to every weight parameters.

    """
    global out
    total_example = X.shape[0]
    small_delta_previous = {}
    weights = reshapeWeights(param, inputUnits, outputUnits, numHiddenLayer, numHiddenUnits)

    # call FeedForward Propagation and Calculate the outputs
    FFModel = Feed(X, weights, HiddenActivation, OutputActivation)

    # add bias to X matrix
    X = mat.c_[mat.ones(X.shape[0]), X]

    # initialize parameters matrix which need to return
    derivatives = mat.zeros(param.shape[0])
    output = FFModel.ActivatedOutputs['output_layer']

    numTrace = 0  # trace variable
    for i in reversed(range(len(weights))):

        if i == numHiddenLayer:
            """
            The loop will enter here when measuring d(j) w.r.t to then weights which are in between last Hidden layer 
            and the Output Layer.
            """
            # calculates small delta values (errors) on each output unit.
            small_delta_output = mat.zeros((X.shape[0], outputUnits))
            for j in range(outputUnits):
                new_output = mat.where(Y == j + 1, 1, 0)
                small_delta_output[:, j] = mat.subtract(mat.c_[output[:, j]], new_output).flatten()

            small_delta_output = mat.multiply(1 / total_example, small_delta_output)

            # Store the delta values which will be useful in next layer calculation (as we are In Backward Direction)
            small_delta_previous.update({"smalldelta": small_delta_output})

            # Compute the Capital delta means multiply to the Input Coming in the Layer Which is Output From
            # Last Hidden Layer.

            delta_output = mat.dot(small_delta_output.transpose(),
                                   FFModel.ActivatedOutputs['Hidden_outputs' + str(i - 1)])

            # regularization
            reg1 = mat.multiply(lamb / total_example,
                                weights['weights' + str(i)][:, 1:weights['weights' + str(i)].shape[1]])

            temp_delta = mat.add(delta_output[:, 1:delta_output.shape[1]], reg1)
            reg_delta_output = mat.c_[delta_output[:, 0], temp_delta]

            # appending the output in derivative in last
            TotalElements = mat.prod(reg_delta_output.shape)

            numTrace = param.shape[0] - TotalElements  # This will help us from which index we should store the values
            # so that These derivative remain in the last of array.

            derivatives[numTrace:param.shape[0]] = reg_delta_output.flatten()

        if 0 < i < numHiddenLayer:
            """ 
            will be enter here only if The Hidden Layer are more than one,means when calculating d(j) w.r.t to the 
            weights between Hidden Layers for ex- if we have two Hidden Layer so loop will enter here only once.
            """
            """
            Now if we are doing backward propagation we need derivative of the Output which going out of the Current 
            Hidden Layer means The Hidden Layer which is on the Right of the weights with respect to which we are 
            Calculation d(j).
            
            So if We are Using ReLu in Hidden Layers we need the Non Activated means we need Linear weights sum
            according to that we are going to Calculate the derivative of ReLu.
            
            On the other hand if we are using Sigmoid in the Hidden Layer we need Activated Outputs Since for 
            Calculating derivative of Sigmoid w.r.t the Linear Linear Equation we sigmoid values of those equation.
            So instead of Calculating those values again and again it is better to Store them once and use them.
            """
            if HiddenActivation == "ReLu":
                out = FFModel.NonActiveOutputs['Hidden_outputs' + str(i)]
            if HiddenActivation == "Sigmoid":
                out = FFModel.ActivatedOutputs['Hidden_outputs' + str(i)]
            derivatives, numTrace = CommonCodeBackProp(i=i,
                                                       OutPut=out,
                                                       Input=FFModel.ActivatedOutputs['Hidden_outputs' + str(i - 1)],
                                                       PreviousDelta=small_delta_previous,
                                                       weights=weights,
                                                       numTrace=numTrace,
                                                       derivatives=derivatives,
                                                       Activation=HiddenActivation,
                                                       totalExample=total_example,
                                                       lamb=lamb
                                                       )

        if i == 0:
            """
             when reached at Input Layer, means measuring d(j) w.r.t weights between Input layer and 1st Hidden layer.
             """
            if HiddenActivation == "ReLu":
                out = FFModel.NonActiveOutputs['Hidden_outputs' + str(i)]
            if HiddenActivation == "Sigmoid":
                out = FFModel.ActivatedOutputs['Hidden_outputs' + str(i)]
            derivatives, numTrace = CommonCodeBackProp(i=i,
                                                       OutPut=out,
                                                       Input=X,
                                                       PreviousDelta=small_delta_previous,
                                                       weights=weights,
                                                       numTrace=numTrace,
                                                       derivatives=derivatives,
                                                       Activation=HiddenActivation,
                                                       totalExample=total_example,
                                                       lamb=lamb
                                                       )

    return derivatives


def CommonCodeBackProp(i, OutPut, Input, PreviousDelta, weights, numTrace, derivatives, Activation, totalExample, lamb):
    """
    This Method contain common code for Computing the BackPropagation for Layer other than output.

    :param Activation: Activation Function For example Sigmoid or ReLu.
    :param i: Loop Number
    :param OutPut: The Output Computed by the layer at which the current loop is for computing Sigmoid derivative of
                   Output. Current Layer is Considered which is on the Right side of The Weights.

    :param Input: The Input which is Coming In the layer at Which the current loop is.
                   Current Layer is Considered which is on the Left side of The Weights.

    :param PreviousDelta: small delta Computed till Previous layer.
    :param weights: Dictionary  Object Containing all reshaped weights.
    :param numTrace: Trace variable.
    :param derivatives: derivative list in which all the derived valued are stored.
    :param totalExample: Total number of example.
    :param lamb: Regularization Parameter.
    :return: Updated numTrace and derivatives list.
    """
    # calculate sigmoid derivative Which is going Out From Current Layer
    Activation_derivative = ActivationDeriavtive.derive(OutPut, Activation)

    # Multiply the Previous Store delta with the Previous weights.
    small_delta_Hidden = mat.dot(PreviousDelta['smalldelta'], weights['weights' + str(i + 1)])

    small_delta_Hidden = mat.multiply(small_delta_Hidden, Activation_derivative)

    # Now small_delta_Hidden that means derivative of J(theta) w.r.t Hidden layer weights also contain extra
    # bias weight which we do not need so we will remove that weight.

    small_delta_Hidden = small_delta_Hidden[:, 1:small_delta_Hidden.shape[1]]

    # update previous delta value
    PreviousDelta.update({"smalldelta": small_delta_Hidden})

    # Compute the Capital delta means multiply to the Input.
    delta_Hidden = mat.dot(small_delta_Hidden.transpose(), Input)

    # regularization
    # extract weights for regularization from Dictionary since we are not going to regularize the bias weights

    parm = weights['weights' + str(i)]
    reg2 = mat.multiply(lamb / totalExample, parm[:, 1:parm.shape[1]])

    temp_delta = mat.add(delta_Hidden[:, 1:delta_Hidden.shape[1]], reg2)
    reg_delta_Hidden = mat.c_[delta_Hidden[:, 0], temp_delta]

    # appending the output in derivative
    TotalElements = mat.prod(reg_delta_Hidden.shape)
    numTrace = numTrace - TotalElements
    derivatives[numTrace:TotalElements + numTrace] = reg_delta_Hidden.flatten()

    return derivatives, numTrace
