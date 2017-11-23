import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os
import time
from glob import glob
from scipy.misc import imsave as ims
from random import randint

import cnn_gan as cg

with tf.Session() as sess:
    batchsize = 64
    iscrop = False
    imagesize = 108
    imageshape = [32, 32, 3]
    z_dim = 100

    gf_dim = 32
    df_dim = 32

    c_dim = 3
    learningrate = 0.0002
    beta1 = 0.5

    ################# MODEL + OPTIMIZER
    # build model
    images = tf.placeholder(tf.float32, [batchsize] + imageshape, name="real_images")
    zin = tf.placeholder(tf.float32, [None, z_dim], name="z")
    
    # Generator
    G = cg.generator(zin)
    
    # Discriminators
    D_prob, D_logit = cg.discriminator(images, df_dim)
    D_fake_prob, D_fake_logit = cg.discriminator(G, df_dim, reuse=True)

    # Losses
    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.ones_like(D_fake_logit)))
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit, tf.ones_like(D_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.zeros_like(D_fake_logit)))
    dloss = d_loss_real + d_loss_fake

    # Optimizer
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    d_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(dloss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(gloss, var_list=g_vars)
    tf.initialize_all_variables().run()

    ########## Data + Train
    
    # Save
    saver = tf.train.Saver(max_to_keep=10)

    data = None
    batch = None
    batch = unpickle("cifar-10-batches-py/data_batch_1")
    
    counter = 1
    start_time = time.time()

    display_z = np.random.uniform(-1, 1, [batchsize, z_dim]).astype(np.float32)

    realfiles = data[0:64]
    realim = [get_image(batch_file, [64,64,3], is_crop=False) for batch_file in realfiles]
    real_img = np.array(realim).astype(np.float32)
    ims("results/imagenet/real.jpg",merge(real_img,[8,8]))

    train = True
    if train:
        # saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        for epoch in xrange(10):
            batch_idx = 30000 if cifar else (len(data)/batchsize)-2
            for idx in xrange(batch_idx):
                batch_images = None

                batchnum = randint(0,150)
                trainingData = batch["data"][batchnum*batchsize:(batchnum+1)*batchsize]
                trainingData = transform(trainingData, is_crop=False)
                batch_images = np.reshape(trainingData,(batchsize,3,32,32))
                batch_images = np.swapaxes(batch_images,1,3)

                batch_z = np.random.uniform(-1, 1, [batchsize, z_dim]).astype(np.float32)

                for k in xrange(1):
                    sess.run([d_optim],feed_dict={ images: batch_images, zin: batch_z })
                for k in xrange(1):
                    sess.run([g_optim],feed_dict={ zin: batch_z })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, " \
                    % (epoch, idx, batch_idx,
                        time.time() - start_time,))

                if counter % 200 == 0:
                    sdata = sess.run([G],feed_dict={ zin: display_z })
                    print np.shape(sdata)
                    ims("results/imagenet/"+str(counter)+".jpg",merge(sdata[0],[8,8]))
                    errD_fake = d_loss_fake.eval({zin: display_z})
                    errD_real = d_loss_real.eval({images: batch_images})
                    errG = gloss.eval({zin: batch_z})
                    print errD_real + errD_fake
                    print errG
                    # print("errd: %4.4f errg: $4")
                if counter % 1000 == 0:
                    saver.save(sess, os.getcwd()+"/training/train",global_step=counter)
    else:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        batch_z = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
        batch_z = np.repeat(batch_z, batchsize, axis=0)
        for i in xrange(z_dim):
            edited = np.copy(batch_z)
            edited[:,i] = (np.arange(0.0, batchsize) / (batchsize/2)) - 1
            sdata = sess.run([G],feed_dict={ zin: edited })
            ims("results/imagenet/"+str(i)+".jpg",merge(sdata[0],[8,8]))