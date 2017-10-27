#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:22:27 2017

@author: hasnat
# Sample implementation of K-means
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

numDim = 2
numClass = 2
numObs = 1000

numIt = 100
eps = 0.0001

X_1 = np.random.normal(0.4, 0.1, (numObs/2, numDim))
X_2 = np.random.normal(0.6, 0.1, (numObs/2, numDim))

X = np.vstack((X_1, X_2))

gtLabels = np.random.randint(0, numClass, numObs)

fig = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.plot(X[:,0], X[:,1], '*')

means = np.random.random((numClass, numDim))
labels = np.zeros((numDim, 1))
tDist = np.zeros((numObs, numClass))
tMean = np.zeros((numClass, numDim))

diffMean = list()
for itc in range(numIt):
    print(str(itc))    
    # Compute distances
    for j in range(numClass):
        tDist[:,j] = np.sqrt(np.sum((X - means[j, :])**2, axis=1))
    
    tLabels = np.argmin(tDist, axis=1)
    
    # Update mean and compute objective score
    objDist = 0
    for j in range(numClass):
        clSamples = X[np.where(tLabels==j)[0],:]
        tMean[j,:] = np.mean(clSamples, axis=0)
        objDist += np.sum(np.sqrt(np.sum((clSamples - tMean[j,:])**2, axis=1)))
    
    diffMean.append(np.sqrt(np.sum((means - tMean)**2)))
    
    if(diffMean[itc] < eps):
        break
    means = tMean.copy()
    
ax2 = plt.subplot2grid((2, 2), (0, 1))
for j in range(numClass):
    clSamples = X[np.where(tLabels==j)[0],:]
    ax2.plot(clSamples[:,0], clSamples[:,1], '*')    

ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax2.plot(diffMean, '-.*')