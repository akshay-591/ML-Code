"""
This file contains method which will read the movie text file and return back in form of array
"""
import numpy as mat


def getList(file_name):
    # first we are going to read the file using built in methods and store them as a string in a list then we wil
    # convert that string into numpy array

    # open file
    file = open(file=file_name)
    # initialize list
    movies_array = []
    for line in file.readlines():
        movies_array.append(line.replace('\n', '')) # append string and replace "\n" with blank space
    file.close() # close the file

    # convert into numpy array
    movies_array = mat.array(movies_array)

    return movies_array
