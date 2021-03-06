"""
Created on Mon Nov 20 15:48:17 2017

@author: hasnat
Generate samples
"""
import numpy as np
import tensorflow as tf
import cl_vae_net as cvn

from utils import *
from scipy.misc import imshow as ims
import mnist_input as mnistip
import util_vmf as uvmf

NUM_CLASSES = 10
CODE_LEN = 20

imHeight = 28
imWidth = 28
numCh = 1

FLAGS = tf.app.flags.FLAGS

restoreFileName = "/home/hasnat/Desktop/mnist_verify/vae_mnist/trlogs/model-mn_j_cl_2.ckpt-2000"

images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, imHeight, imWidth, numCh])
labels = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size])
guessed_z = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, CODE_LEN])
images_ev, labels_ev = mnistip.inputs(eval_data=True, numPrThread=1)

# CL ZONE
_, encoded_, kappaVal, _, t_num_dim = cvn.inference_cl_cnn(images, CODE_LEN, NUM_CLASSES)
im_gen = cvn.generate_cl_cnn_simp(guessed_z, t_num_dim)

wts = [v for v in tf.trainable_variables() if(v.name.lower().find('rec_softmax_linear/') >= 0)]

#####################
saver1 = tf.train.Saver(tf.trainable_variables())    
  
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

saver1.restore(sess, restoreFileName)
  
###########
#wts = np.asarray(sess.run([wts])).squeeze().T
#genCode = wts[6] + (np.random.sample((1,CODE_LEN))-0.5)*0.5
#genCode = np.reshape(genCode, (1,CODE_LEN))

allDigImgs = np.zeros((10, 28, 28))
im_gen_orig= np.zeros((10, 28, 28))
im_gen_mean = np.zeros((10, 28, 28))

ccd = 0



cd = 0
for jj in range(1000):
    #print(jj)
    if(cd>9):
        break;        
    images_, labels = sess.run([images_ev, labels_ev])
    if(labels==cd):
        allDigImgs[cd,:,:] = images_[0, :, :, 0]
        cd = cd + 1
        # print(cd)

for jj in range(10):
    genCode, kappa_ = np.asarray(sess.run([encoded_, kappaVal], feed_dict={images: np.expand_dims(np.expand_dims(allDigImgs[jj,:,:], 0), 3)})).squeeze()
    genCode = np.reshape(genCode, (1,CODE_LEN))
    im_gen_orig[jj,:,:] = np.asarray(sess.run([im_gen], feed_dict={guessed_z: genCode})).squeeze()

wts = np.asarray(sess.run([wts])).squeeze().T
for jj in range(10):
    genCode= np.expand_dims(wts[jj]*kappa_, 0)
    im_gen_mean[jj,:,:] = np.asarray(sess.run([im_gen], feed_dict={guessed_z: genCode})).squeeze()
    
#
#wts = np.asarray(sess.run([wts])).squeeze().T
#
#genCode = wts[labels]*kappa_                
#im_gen_mean = np.asarray(sess.run([im_gen], feed_dict={guessed_z: genCode})).squeeze()
#

im_gen_vmf = np.zeros((80, 28, 28))
tindx = 0
for jj in range(10):
    for kk in range(8):
        vmfsampCode = uvmf.sample_vMF(np.expand_dims(wts[jj, :], 0), 500, 1)*kappa_
        im_gen_vmf[tindx,:,:] = np.asarray(sess.run([im_gen], feed_dict={guessed_z: vmfsampCode})).squeeze()
        tindx = tindx+1

from utils import *
from scipy.misc import imsave as ims
ims("samps_500.jpg",merge(im_gen_vmf,[10,8]))
#from scipy.misc import imshow
#imshow(im_gen)
#imshow(np.asarray(images_).squeeze())

#import matplotlib.pyplot as plt
#plt.subplot(141)
#plt.title('Original')
#plt.imshow(np.asarray(images_).squeeze())
#plt.subplot(142)
#plt.title('Orig-generated')
#plt.imshow(im_gen_orig)
#plt.subplot(143)
#plt.title('Mean-generated')
#plt.imshow(im_gen_mean)
#plt.subplot(144)
#plt.title('vmf_samp-generated')
#plt.imshow(im_gen_vmf)
#plt.show()