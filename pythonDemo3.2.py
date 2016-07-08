#!/usr/bin/env python

"""
    CSV data is brought in which contains two features and one category:
    x data (student's grades on 2 separate tests) and y data
    representing whether they were admitted to university or not.

    Logistic regression is performed on these 2 features, resulting in
    a polynomial best fit boundary estimation between the regions of accepted
    and rejected potential students.
"""

__author__ = "Kyle Laskowski"
__copyright__ = "License CC BY-SA 3.0"

import numpy as np
from scipy.special import expit
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import PolynomialFeatures


def costLogisticRegression(theta, x, y, m, lamb):
    """
    Here, logistic regression is used to calculate the 'cost' or error
    in any attempt to fit a function to this data. The gradient 'grad' is
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

# Adds a row of bias values, and generates polynomial features derived from the original data
xDataPoly = PolynomialFeatures(2).fit_transform(xData)
m, n = np.shape(xDataPoly)

# Initialize theta before it is passed to the optimization function
initialTheta = np.ones(n) / 100.
lamb = 0.000001

# And now to call an optimizer
fit_data = fmin_l_bfgs_b(costLogisticRegression, x0=initialTheta, args=(xDataPoly, yData, m, lamb))

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
graphAdmit = plt.scatter(xDataPoly[trueStudent, 1], xDataPoly[trueStudent, 2], marker='D', c='g', label='Admitted')
graphExclude = plt.scatter(xDataPoly[falseStudent, 1], xDataPoly[falseStudent, 2], marker='x', c='r', label='Not Admitted')

# Generate 50x50 values for student scores to be used by plt.contour
u = np.linspace(min(xDataPoly[:, 1]), max(xDataPoly[:, 1]), 50)
v = np.linspace(min(xDataPoly[:, 1]), max(xDataPoly[:, 1]), 50)

# then store as a 2D array, so it satisfies PolynomialFeature input requirements
u = u.reshape(-1, 1)
v = v.reshape(-1, 1)

# Generate polynomial terms for those scores
polyPrediction = np.zeros(shape=(len(u), len(v)))

# Calculates and stores predicted category values for each u, v pair
for i in range(len(u)):
    for j in range(len(v)):
        tempPoly = PolynomialFeatures(2).fit_transform(np.array((u[i], v[j])).T)
        polyPrediction[i, j] = np.dot(tempPoly, thetaOptimized.reshape(-1,1))

# Plot Boundary
plt.contour(u.flatten(), v.flatten(), polyPrediction, 0)

# Create a legend to show the estimated boundary between students and add it to the graph
fitLegend = mpatches.Patch(color='blue', label='Estimated Boundary')
firstLegend = plt.legend(handles=[fitLegend], loc=1)
ax = plt.gca().add_artist(firstLegend)

plt.legend((graphAdmit, graphExclude), ('Admitted', 'Not Admitted'), numpoints=1, loc='lower left', ncol=3, fontsize=11)
plt.show()