#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
#import matplotlib.pyplot as plt


#iteration is which file it loads, then saves to one higher
iteration = 1
version = 5
training = True

filename = "weatherCleaned3.csv"

outIdx = 15
inIdx = 14
epochs = 500
#rowDelim = 850
rowDelim = 65000

#load data from file
data = np.genfromtxt(filename, delimiter=',')[1:]
#np.random.seed(100)
np.random.shuffle(data)
data = data.astype(float)

#separate training and testing data

if version >= 4:
    trainy = data[:rowDelim, outIdx]
    testy = data[rowDelim:, outIdx]
    trainx = data[:rowDelim, :inIdx]
    testx = data[rowDelim:, :inIdx]

print(testy, trainy)

#normalize data
mean = trainx.mean(axis=0)
std = trainx.std(axis=0)
trainx -= mean
trainx /= std
testx -= mean
testx /= std

#create model
model = Sequential()

if version >= 4:
    model.add(Dense(13, input_dim = 14, activation='tanh'))
    model.add(Dense(10, input_dim = 13, activation='tanh'))
    model.add(Dense(10, input_dim = 11, activation='tanh'))
    model.add(Dense(1, input_dim = 10, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

#load previously trained weights
if iteration > 0:
    model.load_weights('./params'+str(version)+'-'+str(iteration))

#train the model
if training:
    history = model.fit(trainx, trainy, epochs=epochs ,batch_size=100, verbose = 2, validation_data = (testx, testy))
    model.save_weights('./params'+str(version)+'-'+str(iteration+1))

#test the model
output = model.predict(testx)
prediction = np.round(np.squeeze(output), decimals=1)
print("prediction      actual")
for i in range(20):
    print("%10f    %10f" % (prediction[i], testy[i]))

#print(testy)
#print(prediction)

