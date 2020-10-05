# this file contains code for regularised cos and gradient

import numpy as mat
from scipy import optimize


def hypo(x, theta):
    # calculating prediction
    if theta.shape[0] != x.shape[1] and theta.shape[1] == x.shape[1]:
        prediction = mat.dot(x, theta.transpose())
    else:
        prediction = mat.dot(x, theta)
    return prediction


def cal_cost(param, Y, R, num_feature, num_movies, num_users, lamb):
    """
        :param param: param contains both X and theta parameters in 'C' format.
        :param Y: Y matrix contains user Ratings
        :param R: R contains Logical operator where 1 means user rating is known and 0 means user rating is unknown
        :param num_feature: How many types of contents we are dealing with Like Action, Comedy etc
        :param num_movies: number of movies
        :param num_users: number of users
        :param lamb: Regularization parameters
        :return: a flatt matrix contains both X and Theta parameters in 'C' format
        """
    X = mat.reshape(param[0:num_movies * num_feature], (num_movies, num_feature))
    theta = mat.reshape(param[num_movies * num_feature:param.shape[0]], (num_users, num_feature))
    # calculating prediction
    pred = hypo(X, theta)
    value = (1 / 2)
    # calculating error
    error = mat.subtract(pred, Y)
    # calculating square error
    square_error = mat.power(error, 2)
    # final calculation
    j = mat.multiply(square_error, value)

    # since we are not considering Rating which User have rated so We will multiply above value with R matrix
    # element wise it will automatically make the value zero because it contains 0 at those places where ratings are
    # unknown

    j = mat.sum(mat.multiply(j, R))

    # since in Recommender system we are not including intercept term that's why we will Regularized every theta and X
    # values
    regularize_theta = mat.sum(mat.multiply((lamb / 2), mat.dot(theta.transpose(), theta)))
    regularize_X = mat.sum(mat.multiply((lamb / 2), mat.dot(X.transpose(), X)))

    cost = j + regularize_theta + regularize_X
    return j


def cal_grad(param, Y, R, num_feature, num_movies, num_users, lamb):
    """

    :param param: param contains both X and theta parameters in 'C' format since we have to return only a single
                  flatt matrix so that we can work easily with optimization function.
    :param Y: Y matrix contains user Ratings
    :param R: R contains Logical operator where 1 means user rating is known and 0 means user rating is unknown
    :param num_feature: How many type of content we are dealing with Like Action, Comedy etc
    :param num_movies: number of movies
    :param num_users: number of users
    :param lamb: Regularization parameters
    :return: a flatt matrix contains both X and Theta parameteres in 'C'
    """

    # reshape param matrix and get X and theta out

    X = mat.reshape(param[0:num_movies * num_feature], (num_movies, num_feature))
    theta = mat.reshape(param[num_movies * num_feature:param.shape[0]], (num_users, num_feature))

    # calculating prediction
    pred = hypo(X, theta)

    # calculating error
    error = mat.subtract(pred, Y)

    # since we are not considering Rating which User  have not rated so We will multiply above value with R matrix
    # element wise it will automatically make the value zero because it contains 0 at those places where ratings are
    # unknown

    opt = mat.multiply(error, R)

    # calculate cost function w.r.t theta
    opt_theta = mat.add(mat.dot(opt.transpose(), X), mat.multiply(lamb, theta))

    # calculate cost function w.r.t X
    opt_X = mat.add(mat.dot(opt, theta), mat.multiply(lamb, X))

    # zip the both parameters in single flatt array
    parameters = mat.r_[opt_X.flatten(), opt_theta.flatten()]

    return parameters


def optimize_grad(X, Y, lamb):
    initial_theta = mat.zeros(X.shape[1])
    result = optimize.minimize(fun=cal_cost,
                               x0=initial_theta,
                               args=(X, Y, lamb),
                               method='CG',
                               jac=cal_grad,
                               options={'maxiter': 200})
    return result.x
