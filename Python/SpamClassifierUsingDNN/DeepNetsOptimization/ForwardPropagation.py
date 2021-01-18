"""
This file contain method which will Compute the output using Feed forward propagation

"""

import numpy as mat
from DeepNetsOptimization import HypothesisFunc


class Feed:
    def __init__(self, X, weights, HiddenActivation, OutputActivation):
        """
            This Method is used to Execute FeedForward Propagation algorithm.

            :param weights: Dictionary object containing reshaped weights.
            :param HiddenActivation: Hidden Activation is the Activation Type Which user wants to perform in the
                                     Hidden Layers ex - ReLu or Sigmoid.
            :param OutputActivation: OutputActivation is the Activation Type Which user wants to perform in the Output
                                     Layer. ex- Sigmoid and ReLu.
            :param X: input matrix
            :return: Output from output layer
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
        self.ActivatedOutputs = {}
        self.NonActiveOutputs = {}
        for i in range(len(weights) + 1):

            if i == 0:  # for initial layer
                # add bias (ones column) in input or we can say output from initial layer
                X = mat.c_[mat.ones(X.shape[0]), X]

                # Calculate output of Hidden layer0
                NonActive, Activated = HypothesisFunc.predict(X, weights['weights' + str(i)], HiddenActivation)

                self.ActivatedOutputs.update(
                    {'Hidden_outputs' + str(i): mat.c_[mat.ones(Activated.shape[0]), Activated]})

                self.NonActiveOutputs.update(
                    {'Hidden_outputs' + str(i): mat.c_[mat.ones(NonActive.shape[0]), NonActive]})

            elif len(weights) - 1 > i:  # for Hidden layer after 0 till Last Hidden Layer
                po = self.ActivatedOutputs['Hidden_outputs' + str(i - 1)]  # previous output

                NonActive, Activated = HypothesisFunc.predict(po, weights['weights' + str(i)], HiddenActivation)
                self.ActivatedOutputs.update(
                    {'Hidden_outputs' + str(i): mat.c_[mat.ones(Activated.shape[0]), Activated]})
                self.NonActiveOutputs.update(
                    {'Hidden_outputs' + str(i): mat.c_[mat.ones(NonActive.shape[0]), NonActive]})

            elif len(weights) - 1 == i:  # when we reached at Output later Calculate the output Layer

                po = self.ActivatedOutputs['Hidden_outputs' + str(i - 1)]  # previous output

                NonActive, Activated = HypothesisFunc.predict(po, weights['weights' + str(i)], OutputActivation)

                self.ActivatedOutputs.update(
                    {'output_layer':  Activated})
                self.NonActiveOutputs.update({'output_layer': NonActive})


