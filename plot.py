#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


filename = "weatherCleaned3.csv"
outIdx = 15
inIdx = 14

#load data from file
data = np.genfromtxt(filename, delimiter=',')[1:]
data = data.astype(float)

header = ['MinTemp','MaxTemp','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','Rainfall','RainToday','RainTomorrow']


bigbars=set([9,10,13])


rainy = data[np.where(data[:,outIdx]==1)]
notrainy = data[np.where(data[:,outIdx]==0)]

#superimposed histograms
plt.figure(figsize=(8,14))
plt.subplots_adjust(hspace=0.6)

for ii in range(inIdx):
    plt.subplot(7,2,ii+1)
    plt.title(header[ii])
    if ii in bigbars:
        bins = 8
    else:
        bins = 30
    plt.hist(notrainy[:,ii], alpha=0.7, bins=bins, label='not rainy',  color='orange', normed=True)
    plt.hist(rainy[:,ii], alpha=0.5, bins=bins, label='rainy',  color='blue', normed=True)

plt.savefig('/home/dave/testing/ram/plot.jpg')

