#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:06:13 2017

@author: hasnat
Purpose: Implementation of CNN model from caffe-face for MNIST classification.
https://github.com/ydwen/caffe-face/tree/caffe-face/mnist_example
"""

from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, PReLU
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2 # L2-regularisation
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.initializers import RandomNormal, Constant

import numpy as np

def do_conv_act(x, n_op, k_size, st_size, input_shape=None, k_init='glorot_uniform', k_reg=None, bias_use=True, padtype='same'):    
    # Convolution
    if input_shape is not None:
        x = Conv2D(n_op, kernel_size=(k_size, k_size), strides=(st_size, st_size), 
                         padding=padtype, input_shape=input_shape, use_bias=bias_use, 
                         kernel_initializer=k_init, kernel_regularizer=k_reg)(x)
    else:
        x = Conv2D(n_op, kernel_size=(k_size, k_size), strides=(st_size, st_size), 
                         padding=padtype, use_bias=bias_use,
                         kernel_initializer=k_init, kernel_regularizer=k_reg)(x)
    
    # Nonlinear activation
    x = PReLU()(x)
    
    return x

def do_pool(x, sz):
    return MaxPooling2D(pool_size=(sz, sz))(x)

def rn_(stdval):
    return RandomNormal(stdval)

batch_size = 100
nb_classes = 10
epochs = 6

numTrSamp = 60000
# numTrSamp = 500
numSampToLoadOnce = batch_size*200
gl_wd = 0.0005

# input image dimensions
img_rows, img_cols = 28, 28

path_train = 'mnist_train.h5'
X_test = HDF5Matrix('mnist_test.h5', 'X')
Y_test = HDF5Matrix('mnist_test.h5', 'y')
Y_labels = np.asarray(Y_test)    
# input image dimensions
img_rows = X_test.shape[1]
img_cols = X_test.shape[2]

input_shape = (img_rows, img_cols, 1)

X_test = (np.asarray(X_test, dtype='float32') - 127.5) * 0.0078125    
X_test = np.expand_dims(X_test, 3)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# Define Model 
img_input = Input(shape=input_shape)

c1 = do_conv_act(img_input, 32, k_size=5, st_size=1, input_shape=input_shape, k_reg=l2(gl_wd), padtype='same')
c2 = do_conv_act(c1, 32, k_size=5, st_size=1, k_reg=l2(gl_wd), padtype='same')
p1 = do_pool(c2, 2)

c3 = do_conv_act(p1, 64, k_size=5, st_size=1, k_reg=l2(gl_wd), padtype='same')
c4 = do_conv_act(c3, 64, k_size=5, st_size=1, k_reg=l2(gl_wd), padtype='same')
p2 = do_pool(c4, 2)

c5 = do_conv_act(p2, 128, k_size=5, st_size=1, k_reg=l2(gl_wd), padtype='same')
c6 = do_conv_act(c5, 128, k_size=5, st_size=1, k_reg=l2(gl_wd), padtype='same')
p3 = do_pool(c6, 2)

flat = Flatten()(p3)

fc1 = Dense(3, kernel_regularizer=l2(gl_wd), use_bias=False)(flat)
act_1 = PReLU()(fc1)

bn1 = BatchNormalization()(act_1)
#l2_fc1 = Lambda(lambda  x: K.l2_normalize(x, axis=1))(fc1)
#scale_l2 = Lambda(lambda  x: x*1)(l2_fc1)
    
fc_cl = Dense(nb_classes, activation='softmax')(bn1)

model = Model(inputs=img_input, outputs = fc_cl)

# Set the optimizer
sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()


# Start training    
mulsteplr = [8, 12, 16]
mulstepCnt = 0
for j in range(max(mulsteplr)):
#for j in range(epochs):    
    st = 0
    ed = st + numSampToLoadOnce
    
    while st < numTrSamp:
        print('== Global Epoch: '+ str(j) +'  , LR:: '+ str(K.get_value(sgd.lr)) + ' ==')
        if ed>numTrSamp:
            ed = numTrSamp
        
        if(ed-st < batch_size):
            st = ed-1
            continue

        print('Data extracting from big matrix ...')
        X_train = HDF5Matrix(path_train, 'X', start=st, end=ed)
        Y_train = HDF5Matrix(path_train, 'y', start=st, end=ed) 

        # Preprocess data
        X_train = (np.asarray(X_train, dtype='float32') - 127.5) * 0.0078125    
        X_train = np.expand_dims(X_train, 3)
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        
        print('Fitting model ...')
        model.fit(X_train, Y_train, batch_size=batch_size, shuffle='batch', nb_epoch=1, verbose=1, validation_data=(X_test, Y_test))
        
        # update indices
        st = ed
        ed = st + numSampToLoadOnce        
    
    # decrease learning rate
    if(j>=mulsteplr[mulstepCnt]):
        print(str(j) + ':: Decreasing learning rate')
        K.set_value(sgd.lr, 0.8 * K.get_value(sgd.lr))
        mulstepCnt = mulstepCnt + 1
        
    submodel = Model(inputs=img_input, outputs = act_1)        
    points3D = submodel.predict(X_test)
    np.savez('points3D_bn_'+str(j), X=points3D, y=Y_labels)