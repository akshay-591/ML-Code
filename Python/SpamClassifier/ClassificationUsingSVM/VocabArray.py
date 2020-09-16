# This file contains the methods Code for get Vocab List

import numpy as mat

def getVocab():
    Vocab = mat.loadtxt(fname='vocab.txt', dtype='str')
    return Vocab[:, 1:Vocab.shape[1]]
