import tensorflow as tf
import numpy as np
import mnist_input as minp
import vae_encode_decode as ved
import gan_net_mnist as gnm

FLAGS = tf.app.flags.FLAGS
imHeight = 28
imWidth = 28
numCh = 1
CODE_LEN = 100

###################
images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, imHeight, imWidth, numCh])
gen_code = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, CODE_LEN])


input_images = images
df_dim = 16

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
optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss)
#####################

