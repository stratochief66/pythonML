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
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import PolynomialFeatures
from regressionLib import costLogisticRegression

csvInput = np.loadtxt("data/ex2data1.txt", delimiter=",")

# Number of features and categories, as they are imported together
n = np.shape(csvInput)[1]

# yData will be the last column (third) while the indexing runs 0, 1, 2
# in Python, vs starting at 1 in MatLab
xData = csvInput[:, :n - 1]
yData = csvInput[:, n - 1]

# Adds a row of bias values, and generates polynomial features derived from the original data
# The polynomialDimension parameter can be tuned. Set at 2, at most squared features are generated
polynomialDimension = 2
xDataPoly = PolynomialFeatures(polynomialDimension).fit_transform(xData)

# Initialize theta before it is passed to the optimization function
n = np.shape(xDataPoly)[1]
initialTheta = np.ones(n)
regularizationCoefficient = 0.00001

# And now to call an optimizer and store the results
optimizerResults = fmin_l_bfgs_b(costLogisticRegression, x0=initialTheta, args=(xDataPoly, yData, regularizationCoefficient))

# Pick out and store some key values returned by the optimizer
# Including the best values for theta and the gradient
thetaOptimized = optimizerResults[0]
gradientOptimized = optimizerResults[2]['grad']

# Feedback to the user showing how many optimizer cycles were required before convergence
print "fmin_l_bfgs_b required", optimizerResults[2]['funcalls'], "calls in order to converge on an answer."

# Used as a way to separate the two classifications, for display purposes
trueStudent = np.where(yData == 1)
falseStudent = np.where(yData == 0)

# Generates a plot of original student data
plt.title('Student Admittance Prediction Using Logarithmic Regression')
plt.xlabel('Grade on Test A')
plt.ylabel('Grade on Test B')
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
        tempPoly = PolynomialFeatures(polynomialDimension).fit_transform(np.array((u[i], v[j])).T)
        polyPrediction[i, j] = np.dot(tempPoly, thetaOptimized.reshape(-1,1))

# Plot Boundary using the above generated data
plt.contour(u.flatten(), v.flatten(), polyPrediction, 0)

# Create a legend to show the estimated boundary between students and add it to the graph
fitLegend = mpatches.Patch(color='blue', label='Estimated Boundary')
firstLegend = plt.legend(handles=[fitLegend], loc=1)
ax = plt.gca().add_artist(firstLegend)
plt.legend((graphAdmit, graphExclude), ('Admitted', 'Not Admitted'), numpoints=1, loc='lower left', ncol=3, fontsize=11)
plt.show()