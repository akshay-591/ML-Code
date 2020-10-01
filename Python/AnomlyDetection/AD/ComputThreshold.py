""""
This file contains method to compute a threshold which will help in deciding which data is anomaly is which
is not.
"""
import numpy as mat


def compute(y, p):
    """"
    This method computes the threshold based on best F1 score. Parameter for this method is 'y' the actual output
    matrix and 'p' predicted matrix.
    """
    F1 = 0
    best_F1 = 0
    best_threshold = 0

    # calculate step size for loop
    step_size = (mat.max(p) - mat.min(p)) / 1000

    threshold_array = mat.arange(start=mat.min(p), stop=mat.max(p), step=step_size)

    for threshold in threshold_array:
        # since in this model we are detecting Anomaly so positive output is 1 (Anomaly) if the probability of that
        # data point is less then the threshold
        predicted_values = mat.where(p < threshold, 1, 0)

        true_positive = len(mat.where((predicted_values == 1) & (y == 1))[0])
        false_positive = len(mat.where((predicted_values == 1) & (y == 0))[0])
        false_negative = len(mat.where((predicted_values == 0) & (y == 1))[0])

        # calculate precision
        if true_positive == 0:
            precision = 0
            recall = 0
            F1 = 0
        else:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            F1 = (2 * precision * recall) / (precision + recall)

        if F1 > best_F1:
            best_F1 = F1
            best_threshold = threshold

    return best_threshold, best_F1
