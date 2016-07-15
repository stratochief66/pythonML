#!/usr/bin/env python

"""
    This library contains functions specifically created to store re-usable code
    for solving linear or logistic regression problems
"""

__author__ = "Kyle Laskowski"
__copyright__ = "License CC BY-SA 3.0"

import numpy as np
from scipy.special import expit


def costLogisticRegression(theta, x, y, regularizationCoefficient=0):
    """
    Here, logistic regression is used to calculate the 'cost' or error
    in any attempt to fit a function to this data. The gradient 'grad' is
    also calculated. This function can be called by an optimization algorithm,
    which will then home in on better and better fits to the data.
    """

    m = np.shape(x)[0]

    H = expit(np.dot(x, theta))

    cost = (1. / m) * (np.dot(-y.T, np.log(H)) - np.dot((1. - y.T), np.log(1. - H)))
    grad = (1. / m) * np.dot(x.T, (H - y))

    if regularizationCoefficient:
        cost = cost + (regularizationCoefficient / 2 * m) * np.dot(theta, theta.T)
        grad = grad + (regularizationCoefficient / m) * (np.dot(theta, theta.T) - np.square(theta[0]))

    return cost, grad
