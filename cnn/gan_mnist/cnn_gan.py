#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:33:27 2017

@author: hasnat
"""
from utils import *

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')

def discriminator(image, df_dim, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv')) #16x16x64
    h1 = lrelu(d_bn1(conv2d(h0, df_dim, df_dim*2, name='d_h1_conv'))) #8x8x128
    h2 = lrelu(d_bn2(conv2d(h1, df_dim*2, df_dim*4, name='d_h2_conv'))) #4x4x256
    h4 = dense(tf.reshape(h2, [batchsize, -1]), 4*4*df_dim*4, 1, scope='d_h3_lin')
    return tf.nn.sigmoid(h4), h4

def generator(z, gf_dim):
    z2 = dense(z, z_dim, 4*4*gf_dim*4, scope='g_h0_lin')
    h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, gf_dim*4]))) # 4x4x256
    h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [batchsize, 8, 8, gf_dim*2], "g_h1"))) #8x8x128
    h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [batchsize, 16, 16, gf_dim*1], "g_h2"))) #16x16x64
    h4 = conv_transpose(h2, [batchsize, 32, 32, 3], "g_h4")
    return tf.nn.tanh(h4)