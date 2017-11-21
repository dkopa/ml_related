#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:11:40 2017

@author: hasnat
"""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
do_print = True

def _flatten(input_):
    if(do_print):
      print(input_)
      
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(input_, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    
    return reshape, dim

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

def do_fc_norm(scope_name, input_, n_ip, n_op, stdVal=0.01, wt_d=0.0005, isTrain=True):
  if(do_print):
      print(input_)
      
  with tf.variable_scope(scope_name) as scope:      
      weights = _variable_with_weight_decay('weights', shape=[n_ip, n_op],
                                              stddev=stdVal, wd=wt_d)
      weights_norm = tf.nn.l2_normalize(weights,dim=0)
      fc_op = tf.matmul(input_, weights_norm)
              
  return fc_op, weights_norm

def do_conv_act(scope_name, input_, k_size, n_in, n_op, strd = 1, stdVal=0.01, wt_d=0.0, biasVal=0.0, isBias=True):
    if(do_print):
      print(input_)
      
    with tf.variable_scope(scope_name) as scope:        
      # conv_op
      conv_op = do_convolution(input_, k_size, n_in, n_op, strd = strd, std_val=stdVal,
                               w_d=wt_d, bias_init=biasVal, isBias=isBias)

      # Activation: Parametric ReLU
      act_conv = parametric_relu(conv_op)

    return act_conv

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

def do_deconv_act(scope_name, input_, op_shape, k_size, n_in, n_op, strd = 1, stdVal=0.01, wt_d=0.0, biasVal=0.0, isBias=True, isAct=True):
    if(do_print):
      print(input_)
      
    with tf.variable_scope(scope_name) as scope:        
      # conv_op
      deconv_op = do_deconvolution(input_, op_shape, k_size, n_in, n_op, strd = strd, std_val=stdVal,
                               w_d=wt_d, bias_init=biasVal, isBias=isBias)

      if(isAct):
          # Activation: Parametric ReLU
          return parametric_relu(deconv_op)
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

# encoder
def recognition(input_images, code_len):
    h1 = do_conv_act('conv1_1', input_images, 5, 1, 16, strd = 2, stdVal=5e-2)
    h2 = do_conv_act('conv1_2', h1, 5, 16, 32, strd = 2, stdVal=5e-2)
    h2_flat, t_num_dim = _flatten(h2)
    
    w_mean = do_fc('w_mean', h2_flat, t_num_dim, code_len, stdVal=0.04, wt_d=0.004)
    w_stddev = do_fc('w_stddev', h2_flat, t_num_dim, code_len, stdVal=0.04, wt_d=0.004)
    
    return w_mean, w_stddev, t_num_dim

# decoder
def generation(_code, t_num_dim):
    
    z_develop = do_fc('gen_z_matrix_0', _code, 20, t_num_dim, stdVal=0.04, wt_d=0.004)
    z_develop_rs = tf.reshape(z_develop, [FLAGS.batch_size, 7, 7, 32])
    z_matrix = parametric_relu(z_develop_rs)
    
    h1 = do_deconv_act('gen_dconv1_1', z_matrix, [FLAGS.batch_size, 14, 14, 16], 5, 32, 16, strd = 2, stdVal=5e-2)
    h2 = do_deconv_act('gen_dconv1_2', h1, [FLAGS.batch_size, 28, 28, 1], 5, 16, 1, strd = 2, stdVal=5e-2, isAct=False)
    h2 = tf.nn.sigmoid(h2)
    
    return h2

# encoder
def inference_rec_net(input_images, code_len, NUM_CLASSES, toClassify=True):
    h1 = do_conv_act('rec_conv1_1', input_images, 5, 1, 16, strd = 2, stdVal=5e-2)
    h2 = do_conv_act('rec_conv1_2', h1, 5, 16, 32, strd = 2, stdVal=5e-2)
    h2_flat, t_num_dim = _flatten(h2)
    
    fc4 = do_fc('rec_w_mean', h2_flat, t_num_dim, code_len, stdVal=0.04, wt_d=0.004)
        
    # unit normalized
    fc4 = tf.nn.l2_normalize(fc4,dim=1)
    kappa = tf.get_variable('rec_kappa', dtype=tf.float32, initializer=tf.constant(3.16), trainable=False)
    fc4 = tf.multiply(kappa, fc4, name='rec_kappa_mu')    
      
    if toClassify:
      inpNumNeurons = code_len
      softmax_linear, wts = do_fc_norm('rec_softmax_linear', fc4, inpNumNeurons, NUM_CLASSES, stdVal=0.04, wt_d=0.004)
      return softmax_linear, fc4, kappa, wts, t_num_dim
    else:
      return fc4

# encoder
def inference_rec_net_cent_var(input_images, code_len, NUM_CLASSES, toClassify=True):
    h1 = do_conv_act('rec_conv1_1', input_images, 5, 1, 16, strd = 2, stdVal=5e-2)
    h2 = do_conv_act('rec_conv1_2', h1, 5, 16, 32, strd = 2, stdVal=5e-2)
    h2_flat, t_num_dim = _flatten(h2)
    
    fc4 = do_fc('rec_w_mean', h2_flat, t_num_dim, code_len, stdVal=0.04, wt_d=0.004)
    
    # We eventually want this variation parameter to be equivalent to: cent_var = (cent*kappa - fc4), such that the 
    # difference between center and sample is minimized. I think that if this variable is learned properly then we can 
    # generate samples from the learned weights and a noisy version of this variable. Note that, this eventually is similar 
    # to the general VAE
    cent_var = do_fc('gen_cent_var', h2_flat, t_num_dim, code_len, stdVal=0.04, wt_d=0.004, isBN=False) # variation of the sample from center
        
    # unit normalized
    fc4 = tf.nn.l2_normalize(fc4,dim=1)
    kappa = tf.get_variable('rec_kappa', dtype=tf.float32, initializer=tf.constant(3.16), trainable=False)
    fc4 = tf.multiply(kappa, fc4, name='rec_kappa_mu')    
      
    if toClassify:
      inpNumNeurons = code_len
      softmax_linear, wts = do_fc_norm('rec_softmax_linear', fc4, inpNumNeurons, NUM_CLASSES, stdVal=0.04, wt_d=0.004)
      return softmax_linear, fc4, cent_var, kappa, wts, t_num_dim
    else:
      return fc4
  
def inference_cl_cnn(images, CODE_LEN, NUM_CLASSES):
  # conv1-pool1
  conv1_1 = do_conv_act('rec_conv1_1', images, 5, 1, 32, stdVal=5e-2)
  conv1_2 = do_conv_act('rec_conv1_2', conv1_1, 5, 32, 32, stdVal=5e-2)
  pool1 = do_maxpool('rec_pool1', conv1_2, 2, 2)
  
  # conv2-pool2
  conv2_1 = do_conv_act('rec_conv2_1', pool1, 5, 32, 64, stdVal=5e-2)
  conv2_2 = do_conv_act('rec_conv2_2', conv2_1, 5, 64, 64, stdVal=5e-2)
  pool2 = do_maxpool('rec_pool2', conv2_2, 2, 2)

  # conv2-pool2
  conv3_1 = do_conv_act('rec_conv3_1', pool2, 5, 64, 128, stdVal=5e-2)
  conv3_2 = do_conv_act('rec_conv3_2', conv3_1, 5, 128, 128, stdVal=5e-2)
  pool3 = do_maxpool('rec_pool2', conv3_2, 2, 2)
  
  flat_pool, t_num_dim = _flatten(pool3)
                           
  # fc4
  fc4 = do_fc('rec_local4', flat_pool, t_num_dim, CODE_LEN, stdVal=0.04, wt_d=0.004, biasVal=0.1, isBN=False)
  
  # unit normalized
  fc4 = tf.nn.l2_normalize(fc4,dim=1)
  kappa = tf.get_variable('rec_kappa', dtype=tf.float32, initializer=tf.constant(2.0), trainable=False)
  fc4 = tf.multiply(kappa, fc4, name='rec_kappa_mu')
    
  if(do_print):
      print(fc4)
  
  softmax_linear, wts = do_fc_norm('rec_softmax_linear', fc4, CODE_LEN, NUM_CLASSES, stdVal=0.04, wt_d=0.004)
  return softmax_linear, fc4, kappa, wts, t_num_dim
  
# decoder
def generate_cl_cnn(_code, t_num_dim):    
    z_develop = do_fc('gen_z_matrix_0', _code, 20, t_num_dim, stdVal=0.04, wt_d=0.004)
    z_develop_rs = tf.reshape(z_develop, [FLAGS.batch_size, 4, 4, 128])
    z_matrix = parametric_relu(z_develop_rs)

    h1 = do_deconv_act('gen_dconv1_1', z_matrix, [FLAGS.batch_size, 7, 7, 128], 5, 128, 128, strd = 2, stdVal=5e-2)
    h2 = do_deconv_act('gen_dconv1_2', h1, [FLAGS.batch_size, 7, 7, 64], 5, 128, 64, strd = 1, stdVal=5e-2)
    
    h3 = do_deconv_act('gen_dconv2_1', h2, [FLAGS.batch_size, 14, 14, 64], 5, 64, 64, strd = 2, stdVal=5e-2)
    h4 = do_deconv_act('gen_dconv2_2', h3, [FLAGS.batch_size, 14, 14, 32], 5, 64, 32, strd = 1, stdVal=5e-2)
    
    h5 = do_deconv_act('gen_dconv3_1', h4, [FLAGS.batch_size, 28, 28, 32], 5, 32, 32, strd = 2, stdVal=5e-2)
    h6 = do_deconv_act('gen_dconv3_2', h5, [FLAGS.batch_size, 28, 28, 32], 5, 32, 32, strd = 1, stdVal=5e-2)
    
    h7 = do_deconv_act('gen_dconv4_1', h6, [FLAGS.batch_size, 28, 28, 1], 5, 32, 1, strd = 1, stdVal=5e-2, isAct=False)
    im_op = tf.nn.sigmoid(h7)
    return im_op

# decoder
def generate_cl_cnn_simp(_code, t_num_dim):    
    z_develop = do_fc('gen_z_matrix_0', _code, 20, t_num_dim, stdVal=0.04, wt_d=0.004)
    z_develop_rs = tf.reshape(z_develop, [FLAGS.batch_size, 4, 4, 128])
    z_matrix = parametric_relu(z_develop_rs)

    h1 = do_deconv_act('gen_dconv1_1', z_matrix, [FLAGS.batch_size, 7, 7, 64], 5, 128, 64, strd = 2, stdVal=5e-2)    
    h2 = do_deconv_act('gen_dconv2_1', h1, [FLAGS.batch_size, 14, 14, 32], 5, 64, 32, strd = 2, stdVal=5e-2)
    h3 = do_deconv_act('gen_dconv3_1', h2, [FLAGS.batch_size, 28, 28, 1], 5, 32, 1, strd = 2, stdVal=5e-2, isAct=False)
    im_op = tf.nn.sigmoid(h3)
    return im_op

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
