""""
This file contains Recommender System Model and we are going to perform this model on Movies data set.
Total users are = 943 and Total Movies are = 1682. In this Model we are going to predict Unknown rating using
Collaborative filtering algorithm. Data Set is taken from ML course By prof Andrew ng. on Coursera.
"""

import numpy as mat
from scipy import io
from Collaborativefiltering import CostGrad, getMovieList, Normalize
from DebuggingTool import TestFunction

# load data set
data = io.loadmat('../Data/ex8_movies.mat')

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
movie_data = io.loadmat('../Data/ex8_movieParams.mat')

theta = movie_data['Theta']  # weight parameters of all users
X = movie_data['X']  # contain content percentage of the movies

num_feature = movie_data['num_features'].item()  # number of different content like Action, Romantic, Comedy etc.
num_movies = movie_data['num_movies'].item()  # number of movies
num_users = movie_data['num_users'].item()  # number of Users

# ==================== Testing  Gradient using Numerical vs Analytical Technique ==================
print(
    "\n=========Testing Cost function and Gradient using Numerical vs Analytical Technique at lambda=0  =============")
user = 4
movies = 5
feature = 3
param = mat.r_[X[0:movies, 0:feature].flatten(), theta[0:user, 0:feature].flatten()]

cost = CostGrad.cal_cost(param, Y[0:movies, 0:user], R[0:movies, 0:user],
                         feature, movies, user, 0)

print("Initial Cost at lambda 0 is = ", cost)

lamb = 0
TestFunction.checkGrads(lamb)

# ======================= Calculating Cost again at different Regularization value ========================
print("\n========= Testing Cost function and Gradient using Numerical vs Analytical Technique at lambda=1.5 ==========")
lamb = 1.5
cost = CostGrad.cal_cost(param, Y[0:movies, 0:user], R[0:movies, 0:user],
                         feature, movies, user, lamb)
print("\nCost at lambda=1.5 = ", cost)
TestFunction.checkGrads(lamb)

# ================================ Entering new User rating ================================================
print("\n================================ Entering new User rating ================================================\n")
# now to see how it works we are going to add our own rating for movies. First we load the movie list

movie_list = getMovieList.getList(file_name='../Data/movie_ids.txt')

# enter rating
my_ratings = mat.zeros((num_movies, 1))
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

# now we will going to print the rating to see
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(movie_list[i], "rating given by user is ", my_ratings[i].item())

# ===================================== Learning Parameters ===========================================
print("\n===================================== Learning Parameters ===========================================")

# add new user data in Y and R matrix
num_users = num_users+1
new_Y = mat.c_[my_ratings, Y]
new_R = mat.c_[mat.where(my_ratings == 0, 0, 1), R]
# normalize ratings
Ymean, Ynorm = Normalize.normalizeData(new_Y, new_R)

# initialize parameter randomly

X = mat.random.randn(num_movies, num_feature)
theta = mat.random.randn(num_users, num_feature)

initial_param = mat.r_[X.flatten(), theta.flatten()]
lamb = 10

learned_param = CostGrad.optimize_grad(param=initial_param,
                                       maxiter=100,
                                       args=(Ynorm, new_R, num_feature, num_movies, num_users, lamb))

# reshape learned parameters
learned_X = mat.reshape(learned_param[0:num_movies * num_feature], (num_movies, num_feature))
learned_theta = mat.reshape(learned_param[num_movies * num_feature:learned_param.shape[0]], (num_users, num_feature))

# ========================== Recommendation for new user ==================================
print("\n =================== Recommendation for new User is ===================")
# get prediction
prediction = CostGrad.prediction(learned_X, learned_theta)
# add the mean back for the new user only
newUser_prediction = mat.add(mat.c_[prediction[:, 0]], Ymean)

# now sort the predictions in descending order and Recommend only Top 10 predictions or Recommendations
sorted_prediction = mat.sort(newUser_prediction, axis=0)[::-1]

for i in range(10):
    j = mat.where(newUser_prediction == sorted_prediction[i])[0]
    print(movie_list[j], " Predicted rating is ", newUser_prediction[j])

# ==================================== End of Model ==============================
print("\n ==================================== End of Model ==============================")