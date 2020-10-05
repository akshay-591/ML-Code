""""
This file contains Recommender System Model and we are going to perform this model on Movies data set.
Total users are = 943 and Total Movies are = 1682. In this Model we are going to predict Unknown rating using
Collaborative filtering algorithm. Data Set is taken from ML course By prof Andrew ng. on Coursera.
"""

import numpy as mat
from scipy import io
from Collaborativefiltering import Regularised_Linear_Cost_Grad as CostF

# load data set
data = io.loadmat('ex8_movies.mat')

R = data['R']  # this matrix contains Logical operator where 1 means user rated that movies and 0 user not rated
# that movie.

Y = data['Y']  # this matrix contains ratings of every user.

""" 
in both R and Y matrix row represent Movies and Column represents Users so both matrix is of 1682*943 dimension.
"""


"""
now we are only going to work with with those indices which have value = 1 in R(i,j) matrix where j represent user and 
i represent movie because 1 means user give some rating to that movie and 0 means user did not rated that movies and 
we are going to predict some rating for that indices.
"""

# showing average rating of 1st Movie (Toy Story).

# first get the indices from 1st row of R matrix where value is = 1.
ind = mat.where(R[0, :] == 1)[0]

# now get the means of the value at ind in 1st row of Y matrix.
avg = mat.mean(Y[0, mat.ix_(ind)])

print("average rating of Movie Toy Story is ", mat.round(avg, 2), "out of 5")

# load second data
movie_data = io.loadmat('ex8_movieParams.mat')
theta = movie_data['Theta']
X = movie_data['X'] # contain content percentage of the movies

num_feature = movie_data['num_features'] # number of different content like Action, Romantic, Comedy etc
num_movies = movie_data['num_movies'] # number of movies
num_users = movie_data['num_users'] # number of Users

# Checking Cost function with less data for faster Performance

user = 4
movies = 5
feature = 3
param = mat.r_[X[0:movies, 0:feature].flatten(),theta[0:user, 0:feature].flatten()]
cost = CostF.cal_cost(param, Y[0:movies, 0:user],R[0:movies,0:user],
                      feature, movies, user, 0)
print(cost)

