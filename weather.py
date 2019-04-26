#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#versions:
# 3 -> mse nn relu with rainfall output
# 4 -> mse nn s with raintomorrow output

#iteration is which file it loads, then saves to one higher
iteration = 4
version = 4
training = False

filename = "weatherCleaned1.csv"

#load data from file
data = np.genfromtxt(filename, delimiter=',')[1:]
np.random.seed(100)
np.random.shuffle(data)
data = data.astype(float)

#separate training and testing data
trainx = data[:45000, :13]
testx = data[45000:, :13]

if version >= 4:
    trainy = data[:45000, 16]
    testy = data[45000:, 16]

if version <= 3:
    trainy = data[:45000, 13]
    testy = data[45000:, 13]

#normalize data
mean = trainx.mean(axis=0)
std = trainx.std(axis=0)
trainx -= mean
trainx /= std
testx -= mean
testx /= std

#create model
model = Sequential()

if version == 3:
    model.add(Dense(13, input_dim = 13, activation='tanh'))
    model.add(Dense(10, input_dim = 13, activation='tanh'))
    model.add(Dense(10, input_dim = 11, activation='tanh'))
    model.add(Dense(1, input_dim = 10, activation='relu'))
    model.compile(loss='mse', optimizer='adam')

if version == 4:
    model.add(Dense(13, input_dim = 13, activation='tanh'))
    model.add(Dense(10, input_dim = 13, activation='tanh'))
    model.add(Dense(10, input_dim = 11, activation='tanh'))
    model.add(Dense(1, input_dim = 10, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

#load previously trained waits
if iteration > 0:
    model.load_weights('./params'+str(version)+'-'+str(iteration))

#train the model
if training:
    history = model.fit(trainx, trainy, epochs=200 ,batch_size=100, verbose = 2, validation_data = (testx, testy))
    model.save_weights('./params'+str(version)+'-'+str(iteration+1))

#test the model
prediction = model.predict(testx)
#print(prediction[:10])
#print(testy[:10])

print("prediction      actual")
for i in range(20):
    print("%10f    %10f" % (prediction[i], testy[i]))
