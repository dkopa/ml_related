#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:51:48 2017

@author: hasnat
@author: hasnat
MNIST_VAE_vMFML_DEEP_CNN
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
import cl_vae_net as cvn

from utils import *
from scipy.misc import imsave as ims

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/hasnat/Desktop/mnist_verify/vae_mnist/trlogs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 2001,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


INITIAL_LEARNING_RATE = 0.002
NUM_CLASSES = 10
CODE_LEN = 20
FT_DIM = 20

imHeight = 28
imWidth = 28
numCh = 1

saveLogFile = True
isSaveModel = True

netName = 'mn_vae_basic_ft_2'
saveImPrefix = 'dcnn_cl_gen_ft_3'

isModelFT = True
restoreFileName = "/home/hasnat/Desktop/mnist_verify/vae_mnist/trlogs/model-mn_vae_basic.ckpt-2000"

def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    #### For training CNN
    images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, imHeight, imWidth, numCh])
    labels = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size])
    #####
    
    # Get images and labels
    images_tr, labels_tr = mnistip.distorted_inputs(randFlip=True)
    images_ev, labels_ev = mnistip.inputs(eval_data=True, numPrThread=1)
    # images_ev, labels_ev = mnistip.inputs(eval_data=True)
    
    ########################
    # CL ZONE
    # logits_id, local, cent_var, kappaVal, wts, t_num_dim = cvn.inference_cl_cnn(images, CODE_LEN, NUM_CLASSES)
    logits_id, local, kappaVal, wts, t_num_dim = cvn.inference_rec_net(images, CODE_LEN, NUM_CLASSES)
    
    # Losses -CL
    loss_softmax_id = cvn.loss_softmax(logits_id, labels)
    loss_combined = cvn.loss_total()
    
    # Draw new sample
    # weight_maps = tf.gather(tf.transpose(wts), labels)
    guessed_z = local
    # guessed_z = tf.add(cent_var, weight_maps)
    # guessed_z = weight_maps
    
    # Losses - Generation
    im_gen = cvn.generation(guessed_z, t_num_dim)
    
    # Compute Loss Values
    generation_loss = -tf.reduce_sum(images * tf.log(1e-8 + im_gen) + (1-images) * tf.log(1e-8 + 1 - im_gen),[1,2,3])
    # total_loss = tf.reduce_mean(generation_loss)*0.001 + loss_combined
    # total_loss = tf.reduce_mean(generation_loss) + loss_combined                               
    # total_loss = tf.reduce_mean(loss_combined)
    total_loss = tf.reduce_mean(generation_loss)
    
    # Optimize now
    # Apply variable specific learning rate for optimization
    var_rec = [v for v in tf.trainable_variables() if(v.name.lower().find('rec_') >= 0)]            
    var_gen = [v for v in tf.trainable_variables() if(v.name.lower().find('gen_') >= 0)]
    
    opt_rec = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE*0.001)
    opt_gen = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
    
    grads = tf.gradients(total_loss, var_rec + var_gen)
    grads_rec = grads[:len(var_rec)]
    grads_gen = grads[len(var_rec):]
    
    train_rec = opt_rec.apply_gradients(zip(grads_rec, var_rec))
    train_gen = opt_gen.apply_gradients(zip(grads_gen, var_gen))
    train_op = tf.group(train_rec, train_gen)
    
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

    #########################
    visualization, ev_labels = sess.run([images_ev, labels_ev])
    reshaped_vis = np.squeeze(visualization)
    ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
    
    #########################
    for step in xrange(FLAGS.max_steps):
      _images, _labels = sess.run([images_tr, labels_tr])
      
      _, lossSM, lossTot, lossGen = sess.run([train_op, loss_softmax_id, loss_combined, generation_loss], feed_dict={images: _images, labels: _labels})
          
          
      if step % 100 == 0:
          format_str = ('%s: Step %d, GEN-LOSS = %.2f, SM-loss = %.2f, T-loss = %.2f\n')
          print (format_str % (datetime.now(), step, np.mean(lossGen), lossSM, lossTot))     

          # save intermediate results
          generated_test = sess.run(im_gen, feed_dict={images: visualization, labels: ev_labels})
          generated_test = np.squeeze(generated_test)
          ims("results/"+str(step)+'_'+saveImPrefix+".jpg",merge(generated_test[:64],[8,8]))
          
          ########################### 
          # Evaluate test set
          ###########################
          numTestStep = int(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size)
          predictions_id = np.ndarray(shape=(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, mnistip.NUM_CLASSES), dtype = np.float64)
    
          ftVec = np.ndarray(shape=(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,FT_DIM), dtype = np.float64)
          tLabels = np.ndarray(shape=(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL), dtype = np.float64)

      if step % 500 == 0:    
          # Evaluate
          print("====== Evaluating ID classification ========\n")
          for step_ev in xrange(numTestStep):
              _images, _labels = sess.run([images_ev, labels_ev])
                
              stIndx = step_ev*FLAGS.batch_size
              edIndx = (step_ev+1)*FLAGS.batch_size
                         
              _fts, tpred_id  = sess.run([local, logits_id], feed_dict={images: _images})
                
              predictions_id[stIndx:edIndx, :] = np.asarray(tpred_id)
    
              ftVec[stIndx:edIndx, :] = np.asarray(_fts)
              tLabels[stIndx:edIndx] = np.asarray(_labels)
            
          obs_labels = np.argmax(predictions_id, axis=1)
    
          #print(lab.dtype)
          sum_ = np.sum(tLabels==obs_labels)
          acc_id = sum_/float(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
        
          print('================================')        
          format_str = ('%s: Step %d, ID_Acc = %.5f\n')
          print (format_str % (datetime.now(), step, acc_id))
          print('================================')        
          
      if step % 1000 == 0:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model-'+netName+'.ckpt')    
          saver.save(sess, checkpoint_path, global_step=step)
              
def main(argv=None):  # pylint: disable=unused-argument
  print(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()