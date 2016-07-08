#!/usr/bin/env python

"""
    CSV data is brought in which contains two features and one category:
    x data (student's grades on 2 separate tests) and y data
    representing whether they were admitted to university or not.

    Logistic regression is performed on these 2 features, resulting in
    a linear best fit boundary estimation between the regions of accepted
    and rejected potential students.
"""

__author__ = "Kyle Laskowski"
__copyright__ = "License CC BY-SA 3.0"

import numpy as np
from scipy.special import expit
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def costLogisticRegression(theta, x, y, m):
    """
    Here, logistic regression is used to calculate the 'cost' or error
    in any attempt to fit a function to this data. The gradiant 'grad' is
    also calculated. This function can be called by an optimization algorithm,
    which will then hone in on better and better fits to the data.
    """
    yTrans = y.transpose()
    xTrans = x.transpose()

    H = expit(np.dot(x, theta))

    cost = (1. / m) * (np.dot(-yTrans, np.log(H)) - np.dot((1. - yTrans), np.log(1. - H)))
    grad = (1. / m) * np.dot(xTrans, (H - y))

    return cost, grad

csvInput = np.loadtxt("data/ex2data1.txt", delimiter=",")

# To account for python indexes starting at index 0,
# and shape starting at value 1
m, n = np.shape(csvInput)

# n - 1 , as a dimension with n objects ends at index n - 1
xData = csvInput[:, :n - 1]
yData = csvInput[:, n - 1]

# Pad with a column of bias values
xData = np.insert(xData, 0, 1, axis=1)

m, n = np.shape(xData)

# Initialize theta before it is passed to the optimization function
initialTheta = np.ones(n) / 100.

# and now to call an optimizer
fit_data = fmin_l_bfgs_b(costLogisticRegression, x0=initialTheta, args=(xData, yData, m))

thetaOptimized = fit_data[0]
gradOptimized = fit_data[2]['grad']

print "fmin_l_bfgs_b required", fit_data[2]['funcalls'], "calls in order to converge on an answer."

# Used as a way to separate the two classifications, for display purposes
trueStudent = np.where(yData == 1)
falseStudent = np.where(yData == 0)

# Generates a plot of original student data
plt.title('Student Admittance Prediction Using Logarithmic Regression')
plt.xlabel("Grade on Test A")
plt.ylabel("Grade on Test B")
graphAdmit = plt.scatter(xData[trueStudent, 1], xData[trueStudent, 2], marker='D', c='g', label='Admitted')
graphExclude = plt.scatter(xData[falseStudent, 1], xData[falseStudent, 2], marker='x', c='r', label='Not Admitted')

# Sets a range of x values and generates y values based on the previous fitting
xForGraph = np.arange(min(xData[:, 1]), max(xData[:, 1]), 1)
yForGraph = (-1./thetaOptimized[2]) * (thetaOptimized[0] + np.dot(thetaOptimized[1], xForGraph))
graphFit = plt.plot(xForGraph, yForGraph, lw=2)

# Create a legend to show the estimated boundary between students and add it to the graph
fitLegend = mpatches.Patch(color='blue', label='Estimated Boundary')
firstLegend = plt.legend(handles=[fitLegend], loc=1)
ax = plt.gca().add_artist(firstLegend)
plt.legend((graphAdmit, graphExclude), ('Admitted', 'Not Admitted'), numpoints=1, loc='lower left', ncol=3, fontsize=11)
plt.show()