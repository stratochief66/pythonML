#!/usr/bin/env python

"""
    x,y pairs in a CSV format are brought in and plotted, along with a linear
    fit. Data on the X-Axis represents city populations in 10's of thousands.
    Y-Axis data represents annual company profits in those cities.
"""

__author__      = "Kyle Laskowski"
__copyright__   = "License CC BY-SA 3.0"

import numpy as np
import matplotlib.pyplot as plt

csvInput = np.loadtxt("data/ex1data1.txt",delimiter=",")

xData = csvInput[:, 0]
yData = csvInput[:, 1]

slope, intercept = np.polyfit(xData, yData, 1)

xForGraph = np.arange(min(xData), max(xData), 1)

plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")

plt.plot(xForGraph, slope*xForGraph + intercept, 'b-')

plt.title("Scatter plot of training data")
plt.plot(xData, yData, 'ro')

plt.show()
