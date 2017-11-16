#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:28:56 2017

@author: hasnat
SM
"""

import tensorflow as tf
import mnist_input

FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = 10575#mnist_input.NUM_CLASSES
do_print = 0

def do_fc(scope_name, input_, n_ip, n_op, stdVal=0.01, wt_d=0.0, biasVal=0.0, isBias=True, isBN=False, isTrain=True):
  if(do_print):
      print(input_)
      
  with tf.variable_scope(scope_name) as scope:
      
      weights = _variable_with_weight_decay('weights', shape=[n_ip, n_op],
                                              stddev=stdVal, wd=wt_d)
      if(isBias):
          biases = _variable_on_cpu('biases', [n_op], tf.constant_initializer(biasVal))
          fc_op = tf.matmul(input_, weights) + biases
      else:
          fc_op = tf.matmul(input_, weights)
      
      if(isBN):
        # Batch Normalization    
        fc_op = tf.contrib.layers.batch_norm(fc_op, is_training=isTrain, 
                                              scale=True, updates_collections=None,
                                              variables_collections=["BN_NT_VC"])
        
  return fc_op

def do_conv_act(scope_name, input_, k_size, n_in, n_op, stdVal=0.01, wt_d=0.0, biasVal=0.0, isBias=True):
    if(do_print):
      print(input_)
      
    with tf.variable_scope(scope_name) as scope:        
      # conv_op
      conv_op = do_convolution(input_, k_size, n_in, n_op, std_val=stdVal,
                               w_d=wt_d, bias_init=biasVal, isBias=isBias)

      # Activation: Parametric ReLU
      act_conv = parametric_relu(conv_op)

    return act_conv

def do_convolution(input_, k_size, n_in, n_op, std_val=0.01, w_d=0.0, bias_init=0.0, isBias=True):
    kernel = _variable_with_weight_decay('weights',
                                         shape=[k_size, k_size, n_in, n_op],
                                         stddev=std_val,
                                         wd=w_d)
    conv = tf.nn.conv2d(input_, kernel, [1, 1, 1, 1], padding='SAME')

    if(isBias):
        biases = _variable_on_cpu('biases', [n_op], tf.constant_initializer(bias_init))
        conv_op = tf.nn.bias_add(conv, biases)
    else:
        conv_op = conv
    
    return conv_op

def do_maxpool(scope_name, input_, k_size, strd):
    if(do_print):
      print(input_)
      
    with tf.variable_scope(scope_name) as scope:
        poolOp = tf.nn.max_pool(input_, ksize=[1, k_size, k_size, 1],
                         strides=[1, strd, strd, 1], padding='SAME')
    return poolOp

def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0), dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg
  
def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _flatten(input_):
    if(do_print):
      print(input_)
      
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(input_, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    
    return reshape, dim
                         
def inference(images, isTrain=True, toClassify=True, isBN=True):
  # conv1-pool1
  conv1_1 = do_conv_act('conv1_1', images, 5, 1, 32, stdVal=5e-2)
  conv1_2 = do_conv_act('conv1_2', conv1_1, 5, 32, 32, stdVal=5e-2)
  pool1 = do_maxpool('pool1', conv1_2, 2, 2)
  
  # conv2-pool2
  conv2_1 = do_conv_act('conv2_1', pool1, 5, 32, 64, stdVal=5e-2)
  conv2_2 = do_conv_act('conv2_2', conv2_1, 5, 64, 64, stdVal=5e-2)
  pool2 = do_maxpool('pool2', conv2_2, 2, 2)

  # conv2-pool2
  conv3_1 = do_conv_act('conv3_1', pool2, 5, 64, 128, stdVal=5e-2)
  conv3_2 = do_conv_act('conv3_2', conv3_1, 5, 128, 128, stdVal=5e-2)
  pool3 = do_maxpool('pool2', conv3_2, 2, 2)
  
  flat_pool, t_num_dim = _flatten(pool3)
                           
  # fc4
  fc4 = do_fc('local4', flat_pool, t_num_dim, 3, stdVal=0.04, wt_d=0.004, biasVal=0.1, isBN=False)
  
  if(do_print):
      print(fc4)
  
  if toClassify:
      inpNumNeurons = 3
      softmax_linear = do_fc('softmax_linear', fc4, inpNumNeurons, NUM_CLASSES, stdVal=0.04, wt_d=0.004)
      return softmax_linear, fc4
  else:
      return fc4

def loss_total():
  # Calculate the average cross entropy loss across the batch.
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss_softmax(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
          
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
      
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cnt_loss')
  tf.add_to_collection('losses', cross_entropy_mean)

  return cross_entropy_mean
