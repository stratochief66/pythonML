# pythonML
Re-implementing my learnings in machine learning with Python &amp; Python libraries, rather than Matlab/Octave.

Licensed as:
CC BY-SA 3.0
https://creativecommons.org/licenses/by-sa/3.0/


# pythonDemo1

Takes 2D data, plots it and fits it using Python libraries. A very simple "Hello World!" level re-implementation of the first assignment.

[![python_demo_1_figure_1](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_1_figure_1.png?raw=true)](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_1_figure_1.png?raw=true)


# pythonDemo2

CSV data is brought in which contains two features and one category: x data (square footage of houses and number of bedrooms) and y data representing the price of the house in dollars.

This information is placed into data structures, then displayed in a 3D plot. The gradientDecent function is used to fit the data and the final fitted
value for theta is then printed out.

Below is a 2D rendering of the source data. When the code is actually executed, a 3D plot is generated.

[![python_demo_2_figure_1](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_2_figure_1.png?raw=true)](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_2_figure_1.png?raw=true)


# pythonDemo3.1

CSV data is brought in which contains two features and one category: x data (student's grades on 2 separate tests) and y data representing whether they were admitted to university or not.

Logistic regression is performed on these 2 features, resulting in a linear best fit boundary estimation between the regions of accepted and rejected potential students.

Below is a 2D plot showing all student data, each with a symbol representing their acceptance or rejection. Additionally, the linearly fitted line of descrimination is also rendered. This line is generated based on the previously calculated logistic regression.

[![python_demo_3_figure_1](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_3_figure_1.png?raw=true)](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_3_figure_1.png?raw=true)

# pythonDemo3.2

CSV data is brought in which contains two features and one category: x data (student's grades on 2 separate tests) and y data representing whether they were admitted to university or not.

Logistic regression is performed on these 2 features, resulting in a polynomial best fit boundary estimation between the regions of accepted and rejected potential students.

Below is a 2D plot showing all student data, each with a symbol representing their acceptance or rejection. Additionally, the polynomially fitted line of descrimination is also rendered. This line is generated based on the previously calculated logistic regression.

[![python_demo_3_figure_2](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_3_figure_2.png?raw=true)](https://github.com/stratochief66/pythonML/blob/master/figures/python_demo_3_figure_2.png?raw=true)
