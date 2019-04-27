# Rain Predictor

**Dataset:**

Rainfall in australia

https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

**Inputs:**

MinTemp, MaxTemp, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm

**Outputs:**

Amount of rain today, yes/no prediction of rainfall tomorrow

**Activation functions:**

All but the last layer use hyberbolic tangent activation function.

Tanh is good because some inputs may produce opposite effects when numbers go up or down, so output range of -1 to 1 can take advantage of that. 
I tested it against sigmoid and relu, and it performs much better.

The output uses rectified linear because it is a positive number that can go much higher that 1.

**Loss function and optimizer function:**

I found that mean squared error works well, and so does the adam optimizer.

**Data cleaning:**

The original data file is big enough that I could discard a lot of junk and still have over 50k rows of high-quality data to work with.

I used SQL to select only the rows that don't have any null values, and arranged them in a way that puts all the inputs on the left and outputs on the right for easy array slicing.

There were far more non-rainy days than rainy, so I prevented the nn from learning to always predict no rain by limiting the amount of non-rainy days to the amount of rainy days. That improved accuracy a lot.
