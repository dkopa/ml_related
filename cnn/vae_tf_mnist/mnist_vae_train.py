#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:10:13 2017

@author: hasnat
MNIST_VAE
Acknowledgement/Source: http://kvfrans.com/variational-autoencoders-explained/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import os
import time
import math

import numpy as np
from six.moves import xrange
import tensorflow as tf

import mnist_input as mnistip
import vae_encode_decode as ved
from utils import *
from scipy.misc import imsave as ims

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/hasnat/Desktop/vae_mnist/trlogs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

INITIAL_LEARNING_RATE = 0.01
NUM_CLASSES = 10
FT_DIM = 3
CODE_LEN = 20

imHeight = 28
imWidth = 28
numCh = 1

saveLogFile = True
isSaveModel = True

netName = 'mn_vae_basic'

def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    ########################
    # Get images and labels
    images_tr, labels_tr = mnistip.distorted_inputs(randFlip=False)
    images_ev, labels_ev = mnistip.inputs(eval_data=True)
    
    ########################
    # VAE ZONE
    images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, imHeight, imWidth, numCh])
    # vae_code = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, CODE_LEN])
    
    # Define Encoder
    z_mean, z_stddev, t_num_dim = ved.recognition(images, CODE_LEN)
    
    # Draw new sample
    samples = tf.random_normal([FLAGS.batch_size,CODE_LEN],0,1,dtype=tf.float32)
    guessed_z = z_mean + (z_stddev * samples)
    
    # Define Decoder
    im_gen = ved.generation(guessed_z, t_num_dim)
    
    # Compute Loss Values
    generation_loss = -tf.reduce_sum(images * tf.log(1e-8 + im_gen) + (1-images) * tf.log(1e-8 + 1 - im_gen),[1,2,3])
    latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
    total_loss = tf.reduce_mean(generation_loss + latent_loss)
    
    # Optimize now
    train_op = tf.train.AdamOptimizer(0.001).minimize(total_loss)
    
    #####################
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    #########################
    visualization, _ = sess.run([images_ev, labels_ev])
    reshaped_vis = np.squeeze(visualization)
    ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        
    for step in xrange(FLAGS.max_steps):
      _images, _labels = sess.run([images_tr, labels_tr])
      
      _, lossGen, lossLat = sess.run([train_op, generation_loss, latent_loss], feed_dict={images: _images})

      if step % 20 == 0:
          format_str = ('%s: Step %d, GEN-loss = %.2f, LAT-loss = %.2f\n')
          print (format_str % (datetime.now(), step, np.mean(lossGen), np.mean(lossLat)))     
          
          # save intermediate results
          generated_test = sess.run(im_gen, feed_dict={images: visualization})
          generated_test = np.squeeze(generated_test)
          ims("results/"+str(step)+".jpg",merge(generated_test[:64],[8,8]))
                        
def main(argv=None):  # pylint: disable=unused-argument
  print(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


