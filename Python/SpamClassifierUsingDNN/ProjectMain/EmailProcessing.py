"""
this file contains code and methods for processing Email and extracting feature vector
"""

import numpy as mat
import re as regx
import string
from nltk import PorterStemmer
import VocabArray


def process(Email):
    """
    This method will remove the replace some common things in email like HTML script to empty string, or any numeric
    value to string "number" and so that every email contains similarity Then it will be converted into words after that
    if Vocab list will contain those words its index will be added in the word_indices array and in feature matrix array
    1 will be append at those indices and rest of the indices will be equal to 0.

    :param Email: Email will be a stream which have permission to write and read.
    :return: indices of word from vocab
    list which are present in Email and feature vector matrix size of 1*len(Vocab list) and contain 1 at those
    indices which index word are present in email for ex- if email has word "money" and in vocab list that word is
    present at 10th index so in feature vector matrix 1 will be append at 10th index and rest of the value will be 0.
    """
    # convert to lower case
    email = Email.read().lower()
    # strip any HTML
    temp = regx.sub("<.*?>", " ", email)
    # replace numbers for 0-9 with "number"
    temp = regx.sub("[0-9]+", "number", temp)
    # replace Http adress to "httpaddr"
    temp = regx.sub("(http|https)://[^\s]*", "httpaddr", temp)
    # replace email adress with "emailaddr"
    temp = regx.sub("[^\s]+@.*?\s+", "emailaddr", temp)
    # replace currency sign
    temp = regx.sub("[$]+", "dollar", temp)
    temp = regx.sub("[']", " ", temp)
    # ========================== Tokenize Email ===========================
    # temp = regx.sub(">+|:+|#+|[$]+|[.]+|@+|/+|-+|&+|[*]+|[+]+|=+|[]]+|[?]+|[()]+|[{}]+|,+|[']+|<+|_+|;+|%+", "", temp)

    # remove punctuation
    temp = temp.translate(str.maketrans('', '', string.punctuation))

    # split the string in list of words
    tokenized_list = temp.split()
    stemmer = PorterStemmer()
    a = []
    vocab = VocabArray.getVocab()
    extracted_features = mat.zeros((1, len(vocab)))

    i = 0
    print("========================== Processed Email =========================")
    for w in range(len(tokenized_list)):
        if len(tokenized_list[w]) < 1:
            continue

        # stem the word
        word = stemmer.stem(tokenized_list[w])
        print(word, end=" ")
        if i > 20:
            i = 0
            print("\n")
        # get index of the word from vocab list
        indices = mat.where(vocab == word)[0]
        i += 1
        if len(indices) == 0:
            continue

        a.append(indices)
        extracted_features[:, indices] = 1

    word_indices = mat.c_[mat.array(a)]
    print("\n")
    return word_indices, extracted_features
