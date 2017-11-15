#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:09:45 2017

@author: hasnat
"""

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#from itertools import cycle
#import rand_cmap as rcm

def plot_features_3D(ftVec, lab):    
    # strs = [str(x) for x in range(10)]
    strs = ('r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b')
    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')    
    for i in range(10):
        tIndxs = (lab == i)
        ax.scatter(ftVec[tIndxs,0], ftVec[tIndxs,1], ftVec[tIndxs,2], s=5, c=strs[i], marker=filled_markers[i])
        plt.hold(True)
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=15)
    plt.hold(False)
    
    plt.draw()
    #plt.gcf().clear()
    
#cycol = cycle('bgrcmk').next
             
# Read/Load
npzfile = np.load('points3D_5.npz')
X = npzfile['X']
y = npzfile['y']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_features_3D(X,y)