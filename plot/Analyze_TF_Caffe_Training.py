#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:49:17 2017

@author: hasnat
Purpose: Compare log files tensorflow vs caffe training
"""

import numpy as np
import matplotlib.pyplot as plt

def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def parse_file_caffe(fileName):
    tlines = open(fileName).readlines()
    
    # acc = np.float32([tlines[j].split(' ')[-1] for j in range(len(tlines)) if tlines[j].find('Test net output #0: accuracy')>-1])
    # loss = np.float32([tlines[j].split(' ')[-2] for j in range(len(tlines)) if tlines[j].find('Test net output #1: softmax_loss')>-1])
    # stepnum = np.float32([tlines[j].split(' ')[-4][:-1] for j in range(len(tlines)) if tlines[j].find('Testing net (#0)')>-1])
    stepnum, loss = np.float32(zip(*[(tlines[j].split(' ')[-4][:-1], tlines[j].split(' ')[-1][:-1]) for j in range(len(tlines)) if tlines[j].find(', loss')>-1]))
    return stepnum, loss # , acc

def parse_file_tf(fileName):
    tlines = open(fileName).readlines()
    loss = np.float32([tlines[j].split(' ')[7][:-1] for j in range(len(tlines)) if tlines[j].find('ID-loss')>-1])
    stepnum = np.float32([tlines[j].split(' ')[3][:-1] for j in range(len(tlines)) if tlines[j].find('LR=')>-1])

    return stepnum, loss

fileName = '/home/hasnat/Desktop/casia_verify/train_log_res27_vmfml_msc100.txt'
steps_tf, loss_tf = parse_file_tf(fileName)

fileName = '/home/hasnat/Desktop/updates_ecl_face/hist_log_caffe/LT_CM_Resnet_vMFML_MSC_pl100'
steps_cf, loss_cf = parse_file_caffe(fileName)

#
rng = range(0,len(steps_tf)+1,(len(steps_tf)/len(steps_cf)))
rng[-1] = len(steps_tf)-1
   
pl_tf, = plt.plot(steps_tf[rng], loss_tf[rng], '-r.')
pl_cf, = plt.plot(steps_cf, loss_cf, '-g.')

plt.legend([pl_tf, pl_cf], ['Tensorflow', 'Caffe'])
plt.xlabel('Epochs', fontsize=20, weight='bold')
plt.ylabel('Loss', fontsize=20, weight='bold')
plt.title('Training accuracy w.r.t. epochs', fontsize=20)
plt.show()

