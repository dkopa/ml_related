#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:08:35 2017

@author: hasnat
Purpose: Draw the CNN model and visualize with Tensorboard
Usage: 
1. Execute this file with appropriate CNN model *model.py*
2. In terminal, execute: $tensorboard --logdir="logs"
3. In browser, http://127.0.0.1:6006 and go to GRAPH tab
Reference: https://stackoverflow.com/questions/34230613/display-image-of-graph-in-tensorflow
"""
import numpy as np
import tensorflow as tf
import mnist_sm as mnsm

INITIAL_LEARNING_RATE = 0.01
NUM_CLASSES = 10

batch_size = int(100)
imHeight = int(28)
imWidth = int(28)
numCh = int(1)

with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    
    #### For training CNN
    images = tf.constant(np.random.random((batch_size, imHeight, imWidth, numCh)), tf.float32)
    labels = tf.constant(np.random.randint(0,NUM_CLASSES,size=(batch_size)))
    
    # Build a Graph that computes the logits predictions
    logits_id, local = mnsm.inference(images)
    
    # Calculate losses...
    loss_softmax_id = mnsm.loss_softmax(logits_id, labels)
    loss_combined = mnsm.loss_total()
    
    train_op = tf.train.MomentumOptimizer(INITIAL_LEARNING_RATE, 0.9).minimize(loss_combined)
    
    # Create a saver.
    saver = tf.train.Saver()
    
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    
    # Start running operations on the Graph.
    sess = tf.Session()
    sess.run(init)
    writer = tf.summary.FileWriter('logs', sess.graph)
    print(sess.run(train_op))
    writer.close()