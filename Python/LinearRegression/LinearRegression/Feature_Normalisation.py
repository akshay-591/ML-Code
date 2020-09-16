# This file is for Feature normalization

import numpy as mat


def feature_normalisation(x):
    print(x.shape)
    mean = mat.mean(x, axis=0)
    normalize = mat.subtract(x, mean)

    stnd_deviation = mat.std(normalize, axis=0)

    normalized_feature = mat.divide(normalize, stnd_deviation)

    return normalized_feature

