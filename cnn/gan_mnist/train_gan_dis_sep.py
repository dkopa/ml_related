#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:41:52 2017

@author: hasnat
GAN MNIST - Separate Discriminator
"""

import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os
import time
from glob import glob
from scipy.misc import imsave as ims
from random import randint
from datetime import datetime

# import cnn_gan as cg
import mnist_input as mnistip
import gan_net_mnist as gnm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/hasnat/Desktop/gan_verify/trlogs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20001,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


INITIAL_LEARNING_RATE = 0.01
NUM_CLASSES = 10
CODE_LEN = 20
FT_DIM = CODE_LEN
    
imHeight = 28
imWidth = 28
numCh = 1

learningrate = 0.0002
beta1 = 0.5

saveLogFile = False
isSaveModel = False


netName = 'mn_vae_basic'
saveImPrefix = 'wt_kappa'

isModelFT = False
restoreFileName = "/raid5/hasnat/gan_verify/trlogs/model-mn_vae_basic.ckpt-1000"

def train():
  with tf.Graph().as_default():
    z_dim = 100
    fc_dim = 512 # 1568
  
    ################# MODEL + OPTIMIZER
    # Get images and labels
    images_tr, labels_tr = mnistip.distorted_inputs(randFlip=True)
    images_ev, labels_ev = mnistip.inputs(eval_data=True, numPrThread=1)
    
    # build model
    
    #### Placeholders
    images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, imHeight, imWidth, numCh])
    #####
    
    zin = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name="z")
    
    # Generator
    G = gnm.generator(zin)
    
    # Discriminators    
    D_prob_real, D_logit_real = gnm.discriminator(images)
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        pass 
    D_prob_fake, D_logit_fake = gnm.discriminator(G, reuse=True)
        
    # Losses
    dloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    dloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    dloss = dloss_real + dloss_fake

    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))    

        
    # Optimizer
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    
    d_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1)
    g_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1)
    
    grads_d = tf.gradients(dloss, d_vars)
    grads_g = tf.gradients(gloss, g_vars)
    
    with tf.variable_scope(scope):    
        train_d = d_optim.apply_gradients(zip(grads_d, d_vars))
        train_g = g_optim.apply_gradients(zip(grads_g, g_vars))    
    
    ##### Tensorflow training
    #####################
    # Create a saver.
    saver = tf.train.Saver()
    if(isModelFT):
      saver1 = tf.train.Saver(tf.trainable_variables())    
      
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    
    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    
    if(isModelFT):
      saver1.restore(sess, restoreFileName)
      
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    
        
    ############## TRAIN
    visualization, ev_labels = sess.run([images_ev, labels_ev])
    reshaped_vis = np.squeeze(visualization)
    ims("results/real.jpg",merge_gs(reshaped_vis[:49],[7,7]))
    
    display_z = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
    
    for epoch in xrange(1):
        for steps_ in xrange(200001):
            batch_images, _ = sess.run([images_tr, labels_tr])
            
            batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
            
            # _, lossD, lossGen = sess.run([train_op, dloss, gloss], feed_dict={images: batch_images, zin: batch_z})
            for k in xrange(1):
                _, lossD = sess.run([train_d, dloss],feed_dict={ images: batch_images, zin: batch_z })
            for k in xrange(1):
                _, lossG = sess.run([train_g, gloss],feed_dict={ zin: batch_z })
                
            if steps_ % 100 == 0:
                format_str = ('%s: Step %d, D-LOSS = %.2f, G-loss = %.2f\n')
                print (format_str % (datetime.now(), steps_, lossD, lossG))    
          
            if steps_ % 200 == 0:
                imgGen = sess.run([G], feed_dict={zin: display_z})            
                imgGen = np.squeeze(imgGen)
                ims("results/"+str(steps_)+".jpg",merge_gs(imgGen[0:49],[7,7]))
            
            if steps_ % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model-'+netName+'.ckpt')    
                saver.save(sess, checkpoint_path, global_step=steps_)
                
def main(argv=None):  # pylint: disable=unused-argument
  print(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
