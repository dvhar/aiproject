#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pprint import pprint
from mymodel import mymodel
import matplotlib.pyplot as plt



#iteration is which file it loads, then saves to one higher if training
iteration = 6

#9 - main version
#10 - linear nn
#11 - linear regression
#12 - logistic regression
#13 - main version trained on single location
version = 9


#what to do
training = False
printweights = False
runprediction = True
plotting = True

filename = "weatherCleaned4.csv"
outIdx = 16
inIdx = 16

#testing
#epochs = 500
#full training
epochs = 25000
rowDelim = 50000

#load data from file
data = np.genfromtxt(filename, delimiter=',')[1:]
data = data[:,[2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,23]]

np.random.seed(100)
np.random.shuffle(data)
data = data.astype(float)


#separate training and testing data
trainy = data[:rowDelim, outIdx]
testy = data[rowDelim:, outIdx]
trainx = data[:rowDelim, :inIdx]
testx = data[rowDelim:, :inIdx]

#normalize data
mean = trainx.mean(axis=0)
std = trainx.std(axis=0)
trainx -= mean
trainx /= std
testx -= mean
testx /= std


if printweights:
    print('mean = ')
    pprint(mean)
    print('std = ')
    pprint(std)

#create model
model = Sequential()

if version == 9:
    model.add(Dense(12, input_dim = 16, activation='tanh'))
    model.add(Dense(10, input_dim = 12, activation='tanh'))
    model.add(Dense(10, input_dim = 10, activation='tanh'))
    model.add(Dense(1, input_dim = 10, activation='sigmoid'))
if version == 10:
    model.add(Dense(12, input_dim = 16, activation='linear'))
    model.add(Dense(10, input_dim = 12, activation='linear'))
    model.add(Dense(10, input_dim = 10, activation='linear'))
    model.add(Dense(1, input_dim = 10, activation='linear'))
if version == 11:
    model.add(Dense(1, input_dim = 16, activation='linear'))
if version == 12:
    model.add(Dense(1, input_dim = 16, activation='sigmoid'))

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
    koutput = np.squeeze(model.predict(testx))
    myoutput = np.squeeze([mymodel(x) for x in testx])
    print("kprediction      actual        myprediction")
    for i in range(20):
        print("%10f    %10f     %10f" % (koutput[i], testy[i], myoutput[i]))
    print('keras mean squared error:  ',np.mean((koutput-testy)**2))
    print('mymodel mean squared error:',np.mean((myoutput-testy)**2))

    if plotting:
        rainy = koutput[np.where(testy==1)]
        notrainy = koutput[np.where(testy==0)]
        print(rainy)
        print(notrainy)
        plt.figure(figsize=(4,3.5))
        bins = 30
        plt.hist(notrainy, alpha=0.7, bins=bins, label='not rainy',  color='orange', normed=True)
        plt.hist(rainy, alpha=0.5, bins=bins, label='rainy',  color='blue', normed=True)
        plt.show()

