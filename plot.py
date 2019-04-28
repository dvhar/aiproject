#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


filename = "weatherCleaned4.csv"
outIdx = 16
#inIdx = 14

#load data from file
data = np.genfromtxt(filename, delimiter=',')[1:]
data = data.astype(float)
data = data[:,[2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,23]]

header = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RISK_MM','RainTomorrow']

bigbars=set([2,12,13])


rainy = data[np.where(data[:,outIdx]==1)]
notrainy = data[np.where(data[:,outIdx]==0)]


#superimposed histograms
plt.figure(figsize=(8,24))
plt.subplots_adjust(hspace=0.6)

for ii in range(len(data[0])-1):
    plt.subplot(9,2,ii+1)
    plt.title(header[ii])
    bins = (30,8)[ii in bigbars]
    plt.hist(notrainy[:,ii], alpha=0.7, bins=bins, label='not rainy',  color='orange', normed=True)
    plt.hist(rainy[:,ii], alpha=0.5, bins=bins, label='rainy',  color='blue', normed=True)

plt.savefig('/home/dave/testing/ram/plot.jpg')
#plt.show()

