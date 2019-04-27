# Rain Predictor

**Dataset:**

Rainfall in australia

https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

**Inputs:**

MinTemp, MaxTemp, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, Rainfall


**Output:**

probability of rainfall tomorrow

**Activation functions:**

All but the last layer use hyberbolic tangent activation function.

Tanh is good because some inputs may produce opposite effects when numbers go up or down, so output range of -1 to 1 can take advantage of that. 
I tested it against sigmoid and relu, and it performs much better.

The output uses sigmoid because the output is yes/no, so it needs a function that goes from 0 to 1

**Loss function and optimizer function:**

I found that mean squared error works well, and so does the adam optimizer.

**Data cleaning:**

The original data file is big enough that I could discard a lot of junk, so I used SQL to select only the rows that don't have any null values, and arranged them in a way that puts all the inputs on the left and outputs on the right for easy array slicing.

It gets much better accuracy when a neural net is specific to one city than when it's train on all of them, so I created a csv file with just Albury. Another option would be to add 36 yes-no columns for the 36 different cities.
