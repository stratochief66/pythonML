#!/usr/bin/env python

"""
    CSV data is brought in which contains two features and one category:
    x data (square footage of houses and number of bedrooms)
    and y data representing the price of the house in dollars
    This information is placed into data structures, then displayed in a 3D plot.
    The gradientDecent function is used to fit the data and the final fitted
    value for theta is then printed out.
"""

__author__      = "Kyle Laskowski"
__copyright__   = "License CC BY-SA 3.0"

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csvInput = np.loadtxt("data/ex1data2.txt", delimiter=",")

# To account for python indexes starting at index 0,
# and shape starting at value 1
m, n = np.shape(csvInput)
n = n - 1

xData = csvInput[:, :n]
yData = csvInput[:, n]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xData[:, 0], xData[:, 1], yData, c='r')

ax.set_xlabel('Square Feet')
ax.set_ylabel('Number Of Rooms')
ax.set_zlabel('Selling Price')
plt.show()

m, n = np.shape(xData)

# Normalize data
sigma = np.zeros(n)
mu = np.zeros(n)

# First, identify and store the mean and standard deviations of the data
for i in range(0, n) :
    sigma[i] = xData[:, i].mean()
    mu[i] = xData[:, i].std()
# Next, normalize the data using the above identified features of the data
xData = (xData - mu) / sigma

# Pad with a column of bias values (ones)
xData = np.insert(xData, 0, 1, axis=1)

m, n = np.shape(xData)

# from:
# http://stackoverflow.com/questions/17784587/gradient-descent-using-pyDatathon-and-numpyData
def gradientDescent(xData, yData, theta, alpha, m, numIterations):
    xTrans = xData.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(xData, theta)
        loss = hypothesis - yData
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        """
        Uncommenting the below line allows you to see the cost as it changes
	with each iteration of gradientDecent
        print("Iteration %d | Cost: %f" % (i, cost))
        """
        # average gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update with the new estimation of theta
        theta = theta - alpha * gradient
    return theta, hypothesis

numIterations = 10000
alpha = 0.1

theta = np.ones(n)
theta, hypothesis = gradientDescent(xData, yData, theta, alpha, m, numIterations)
print("The final derived value for theta is:")
print(theta)

"""
# A test to see the data set after normalization
plt.plot(xData[:, 1], xData[:, 2], 'ro')
plt.show()

# Compares the model to reality one on one
print("")
print("House one costs %d" % (yData[4]))
print("")
print("Predicted price is %d" % (hypothesis[4]))
print("")
"""