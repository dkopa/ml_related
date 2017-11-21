#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:34:03 2017

@author: hasnat
"""
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm 

import numpy as np
import util_vmf as uvmf

n = 3
kappa = 10

def plot_features_3D(ftVec, lab, n):    
    color=iter(cm.rainbow(np.linspace(0,1,n)))
    
    # strs = [str(x) for x in range(10)]
    strs = ('r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b')
    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')    
    for i in range(n):
        c=next(color)
        tIndxs = np.where(lab == i)
        ax.scatter(ftVec[tIndxs,0], ftVec[tIndxs,1], ftVec[tIndxs,2], s=5, c=c, marker=filled_markers[i])
        plt.hold(True)
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=15)
    plt.hold(False)    
    
# np.savez(netName+'_'+str(step), X=ftVec, y=tLabels)

numCl=3
numSampPerClass = 100
samps = np.zeros((numSampPerClass*numCl, n))
labels = np.zeros((numSampPerClass*numCl, 1))

for cl in range(numCl):
    direction = np.random.random((1, n))
    mu = np.asarray(direction / np.linalg.norm(direction))
    
    sss = uvmf.sample_vMF(mu, kappa, numSampPerClass)
    samps[cl*numSampPerClass:(cl+1)*numSampPerClass,:] = sss
    labels[cl*numSampPerClass:(cl+1)*numSampPerClass,0] = np.ones(numSampPerClass)*cl


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')    
plot_features_3D(samps,labels, numCl)
    
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')    
#ax.scatter(sss[:,0], sss[:,1], sss[:,2], s=5)
#plt.draw()    
#plt.show()