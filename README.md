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

**Data cleaning:**

There were a ton of null values in the data, so I used SQL to make a new file and cut out all the rows that were causing trouble.

It gets much better accuracy when a neural net is specific to one city than when it's train on all of them, so I created a csv file with just Albury. Another option would be to add 36 yes-no columns for the 36 different cities.
