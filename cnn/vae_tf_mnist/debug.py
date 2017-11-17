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


# Get images and labels
images_tr, labels_tr = mnistip.distorted_inputs(randFlip=True)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

_images, _labels = sess.run([images_tr, labels_tr])        
reshaped_vis = np.squeeze(_images)
ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
    
print(_images.shape)
