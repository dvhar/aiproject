#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pprint import pprint
import matplotlib.pyplot as plt


#iteration is which file it loads, then saves to one higher
iteration = 1
version = 7

#what to do
training = False
printweights = False
runprediction = True

filename = "weatherCleaned3.csv"
outIdx = 15
inIdx = 14
inIdx2 = np.array([1,2,5,6,7,8,9,10,12,13])
epochs = 500
#rowDelim = 850
rowDelim = 65000

#load data from file
data = np.genfromtxt(filename, delimiter=',')[1:]
#np.random.seed(100)
np.random.shuffle(data)
data = data.astype(float)


#separate training and testing data
trainy = data[:rowDelim, outIdx]
testy = data[rowDelim:, outIdx]
if version <= 5:
    trainx = data[:rowDelim, :inIdx]
    testx = data[rowDelim:, :inIdx]
else:
    trainx = data[:rowDelim, inIdx2]
    testx = data[rowDelim:, inIdx2]

#normalize data
mean = trainx.mean(axis=0)
std = trainx.std(axis=0)
trainx -= mean
trainx /= std
testx -= mean
testx /= std

#create model
model = Sequential()

if version <= 5:
    model.add(Dense(13, input_dim = 14, activation='tanh'))
    model.add(Dense(10, input_dim = 13, activation='tanh'))
    model.add(Dense(10, input_dim = 10, activation='tanh'))
    model.add(Dense(1, input_dim = 10, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

else:
    model.add(Dense(12, input_dim = 10, activation='tanh'))
    model.add(Dense(10, input_dim = 12, activation='tanh'))
    model.add(Dense(5, input_dim = 10, activation='tanh'))
    model.add(Dense(1, input_dim = 5, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

#load previously trained weights
if iteration > 0:
    model.load_weights('./params'+str(version)+'-'+str(iteration))

#train the model
if training:
    history = model.fit(trainx, trainy, epochs=epochs ,batch_size=100, verbose = 2, validation_data = (testx, testy))
    model.save_weights('./params'+str(version)+'-'+str(iteration+1))

#print weights and biases for mymodel
if printweights:
    for layer in model.layers:
        print('weights = ')
        pprint(layer.get_weights()[0])
        print('biases = ')
        pprint(layer.get_weights()[1])

#test the model
if runprediction:
    output = model.predict(testx)
    prediction = np.round(np.squeeze(output), decimals=1)
    print("prediction      actual")
    for i in range(20):
        print("%10f    %10f" % (prediction[i], testy[i]))
