# this file contains code and methods for processing Email and extracting feature vector

import numpy as mat
import re as regx
import string
from nltk import PorterStemmer
from ClassificationUsingSVM import VocabArray


def process(Email):
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
        extracted_features[:,indices] = 1

    word_indices = mat.c_[mat.array(a)]
    print("\n")
    return word_indices, extracted_features
