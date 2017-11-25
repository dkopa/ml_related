#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:46:58 2017

@author: hasnat
GAN MNIST - separate discriminator
"""
import tensorflow as tf
from ops import *
FLAGS = tf.app.flags.FLAGS
do_print = True

def _flatten(input_, batch_size=FLAGS.batch_size):
    if(do_print):
      print(input_)
      
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(input_, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    
    return reshape, dim

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0), dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg
  
def do_fc(scope_name, input_, n_ip, n_op, stdVal=0.01, wt_d=0.0, biasVal=0.0, isBias=True, isBN=False, isTrain=True):
  if(do_print):
      print(input_)
      
  with tf.variable_scope(scope_name) as scope:      
      weights = _variable_with_weight_decay('weights', shape=[n_ip, n_op],
                                              stddev=stdVal, wd=wt_d)
      print(weights)
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

def do_conv_act(scope_name, input_, k_size, n_in, n_op, strd = 1, stdVal=0.01, wt_d=0.0, biasVal=0.0, isTrain=True, isBias=True, isBN=False):
    if(do_print):
      print(input_)
      
    with tf.variable_scope(scope_name) as scope:        
      # conv_op
      conv_op = do_convolution(input_, k_size, n_in, n_op, strd = strd, std_val=stdVal,
                               w_d=wt_d, bias_init=biasVal, isBias=isBias)
      
      if(isBN):
        # Batch Normalization    
        conv_op = tf.contrib.layers.batch_norm(conv_op, is_training=isTrain, 
                                              scale=True, updates_collections=None,
                                              variables_collections=["BN_NT_VC"])
        
      # Activation: Parametric ReLU
      # conv_op = parametric_relu(conv_op)
      conv_op = lrelu(conv_op)

    return conv_op

def do_convolution(input_, k_size, n_in, n_op, strd = 1, std_val=0.01, w_d=0.0, bias_init=0.0, isBias=True):
    kernel = _variable_with_weight_decay('weights',
                                         shape=[k_size, k_size, n_in, n_op],
                                         stddev=std_val,
                                         wd=w_d)
    conv = tf.nn.conv2d(input_, kernel, [1, strd, strd, 1], padding='SAME')

    if(isBias):
        biases = _variable_on_cpu('biases', [n_op], tf.constant_initializer(bias_init))
        conv_op = tf.nn.bias_add(conv, biases)
    else:
        conv_op = conv
    
    return conv_op

def do_deconv_act(scope_name, input_, op_shape, k_size, n_in, n_op, strd = 1, stdVal=0.01, wt_d=0.0, biasVal=0.0, isTrain=True, isBias=True, isAct=True, isBN=False):
    if(do_print):
      print(input_)
      
    with tf.variable_scope(scope_name) as scope:        
      # conv_op
      deconv_op = do_deconvolution(input_, op_shape, k_size, n_in, n_op, strd = strd, std_val=stdVal,
                               w_d=wt_d, bias_init=biasVal, isBias=isBias)

      if(isBN):
        # Batch Normalization    
        deconv_op = tf.contrib.layers.batch_norm(deconv_op, is_training=isTrain, 
                                              scale=True, updates_collections=None,
                                              variables_collections=["BN_NT_VC"])
      if(isAct):
          # Activation: Parametric ReLU
          # return parametric_relu(deconv_op)
          return tf.nn.relu(deconv_op)
      else:
          return deconv_op

def do_deconvolution(input_, op_shape, k_size, n_in, n_op, strd = 1, std_val=0.01, w_d=0.0, bias_init=0.0, isBias=True):
    kernel = _variable_with_weight_decay('weights',
                                         shape=[k_size, k_size, n_op, n_in],
                                         stddev=std_val,
                                         wd=w_d)
    conv = tf.nn.conv2d_transpose(input_, kernel, op_shape, [1, strd, strd, 1])

    if(isBias):
        biases = _variable_on_cpu('biases', [n_op], tf.constant_initializer(bias_init))
        deconv_op = tf.nn.bias_add(conv, biases)
    else:
        deconv_op = conv
    
    return deconv_op

def do_maxpool(scope_name, input_, k_size, strd):
    if(do_print):
      print(input_)      
    with tf.variable_scope(scope_name) as scope:
        poolOp = tf.nn.max_pool(input_, ksize=[1, k_size, k_size, 1],
                         strides=[1, strd, strd, 1], padding='SAME')
    return poolOp

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

#def discriminator(images, reuse=False):
#    if reuse:
#        tf.get_variable_scope().reuse_variables()
#
#    h1 = lrelu(conv2d(images, 1, 16, name='d_h0_conv_')) #14x14x16
#    h2 = lrelu(conv2d_bn(h1, 16, 32, name='d_h1_conv')) #7x7x32
#    h3 = lrelu(conv2d_bn(h2, 32, 64, name='d_h2_conv')) #4x4x64
#    h4 = dense(tf.reshape(h3, [FLAGS.batch_size, -1]), 4*4*64, 1, scope='d_h3_lin')
#    return tf.nn.sigmoid(h4), h4

# Discriminator
def discriminator(input_images, batch_size=FLAGS.batch_size, reuse=False, isTrain=True):
    if reuse:
        tf.get_variable_scope().reuse_variables()        
    h1 = do_conv_act('d_conv1_1', input_images, 5, 1, 16, strd = 2, stdVal=2e-2, isTrain=isTrain)
    h2 = do_conv_act('d_conv1_2', h1, 5, 16, 32, strd = 2, stdVal=2e-2, isBN=True, isTrain=isTrain)
    h3 = do_conv_act('d_conv1_3', h2, 5, 32, 64, strd = 2, stdVal=2e-2, isBN=True, isTrain=isTrain)
    h3_flat, t_num_dim = _flatten(h3, batch_size=batch_size)
    
    fc3 = do_fc('d_fc', h3_flat, t_num_dim, 1, stdVal=0.02)
    
    prob_cl = tf.nn.sigmoid(fc3)
    return prob_cl, fc3

#def generator(z, z_dim=100):
#    z2 = dense(z, z_dim, 4*4*64, scope='g_h0_lin')
#    rs = tf.reshape(z2, [-1, 4, 4, 64])
#    rs_bn = tf.contrib.layers.batch_norm(rs, is_training=True, 
#                                              scale=True, updates_collections=None,
#                                              variables_collections=["BN_NT_VC"])
#    h0 = tf.nn.relu(rs_bn) # 4x4x64
#    h1 = tf.nn.relu(conv_transpose_bn(h0, [FLAGS.batch_size, 7, 7, 32], "g_h1_")) #7x7x128
#    h2 = tf.nn.relu(conv_transpose_bn(h1, [FLAGS.batch_size, 14, 14, 16], "g_h2")) #16x16x64
#    h4 = conv_transpose(h2, [FLAGS.batch_size, 28, 28, 1], "g_h4")
#    return tf.nn.sigmoid(h4)

# Generator
def generator(_code, t_num_dim=512, code_len=100, isTrain=True):    
    z_develop = do_fc('g_z_matrix_0', _code, code_len, t_num_dim, stdVal=0.02, isTrain=isTrain, isBN=True)
    z_develop_rs = tf.reshape(z_develop, [FLAGS.batch_size, 4, 4, 32])
    z_matrix = tf.nn.relu(z_develop_rs)
    
    h1 = do_deconv_act('g_dconv1_1', z_matrix, [FLAGS.batch_size, 7, 7, 32], 5, 32, 32, strd = 2, stdVal=5e-2, isBN=True, isBias=False, isTrain=isTrain)
    h2 = do_deconv_act('g_dconv1_2', h1, [FLAGS.batch_size, 14, 14, 16], 5, 32, 16, strd = 2, stdVal=5e-2, isBN=True, isBias=False, isTrain=isTrain)
    h3 = do_deconv_act('g_dconv1_3', h2, [FLAGS.batch_size, 28, 28, 1], 5, 16, 1, strd = 2, stdVal=5e-2, isAct=False, isBN=False, isBias=False, isTrain=isTrain)
    h3 = tf.nn.sigmoid(h3)

    return h3
## Generator
#def generator(_code, t_num_dim=512, code_len=100, isTrain=True, batch_size=FLAGS.batch_size):    
#    z_develop = do_fc('g_z_matrix_0', _code, code_len, t_num_dim, stdVal=0.02, isTrain=isTrain, isBN=True)
#    z_develop_rs = tf.reshape(z_develop, [batch_size, 4, 4, 32])
#    z_matrix = tf.nn.relu(z_develop_rs)
#    
#    h1 = do_deconv_act('g_dconv1_1', z_matrix, [batch_size, 7, 7, 32], 5, 32, 32, strd = 2, stdVal=5e-2, isBN=True, isBias=False, isTrain=isTrain)
#    h2 = do_deconv_act('g_dconv1_2', h1, [batch_size, 14, 14, 16], 5, 32, 16, strd = 2, stdVal=5e-2, isBN=True, isBias=False, isTrain=isTrain)
#    h3 = do_deconv_act('g_dconv1_3', h2, [batch_size, 28, 28, 1], 5, 16, 1, strd = 2, stdVal=5e-2, isAct=False, isBias=False, isTrain=isTrain)
#    h3 = tf.nn.sigmoid(h3)
#
#    return h3
  
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
