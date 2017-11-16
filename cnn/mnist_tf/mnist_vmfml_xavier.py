import tensorflow as tf
import mnist_input

FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = mnist_input.NUM_CLASSES

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

def do_fc_norm(scope_name, input_, n_ip, n_op, stdVal=0.01, wt_d=0.0, biasVal=0.0, isBias=True, isBN=False, isTrain=True):
  if(do_print):
      print(input_)
      
  with tf.variable_scope(scope_name) as scope:
      weights = _variable_with_weight_decay('weights', shape=[n_ip, n_op], wd=wt_d)
      weights_norm = tf.nn.l2_normalize(weights,dim=0)
      fc_op = tf.matmul(input_, weights_norm)              
  return fc_op

def do_fc(scope_name, input_, n_ip, n_op, stdVal=0.01, wt_d=0.0, biasVal=0.0, isBias=True, isBN=False, isTrain=True):
  if(do_print):
      print(input_)
      
  with tf.variable_scope(scope_name) as scope:
      weights = _variable_with_weight_decay('weights', shape=[n_ip, n_op], wd=wt_d)
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

def do_convolution(input_, k_size, n_in, n_op, w_d=0.0):
    kernel = _variable_with_weight_decay('weights', shape=[k_size, k_size, n_in, n_op], wd=w_d)
    print(input_)
    conv = tf.nn.conv2d(input_, kernel, [1, 1, 1, 1], padding='SAME')
    print(conv)
    
    biases = _variable_on_cpu('biases', [n_op], tf.contrib.layers.xavier_initializer())
    conv_op = tf.nn.bias_add(conv, biases)
    
    return conv_op

def do_conv_act(scope_name, input_, k_size, n_in, n_op, wt_d=0.0):
  with tf.variable_scope(scope_name) as scope:
      # conv_op
      conv_op = do_convolution(input_, k_size, n_in, n_op, w_d=wt_d)    

      # Activation: Parametric ReLU
      act_conv = parametric_relu(conv_op)            
  return act_conv

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def compute_cnn_features(images, isTrain=True, toClassify=True):
    # conv-pool-1
    conv1_1 = do_conv_act('conv1_1', images, 5, 1, 32)
    conv1_2 = do_conv_act('conv1_2', conv1_1, 5, 32, 32)
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    
    # conv-pool-2
    conv2_1 = do_conv_act('conv2_1', pool1, 5, 32, 64)
    conv2_2 = do_conv_act('conv2_2', conv2_1, 5, 64, 64)
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
    # conv-pool-3
    conv3_1 = do_conv_act('conv3_1', pool2, 5, 64, 128)
    conv3_2 = do_conv_act('conv3_2', conv3_1, 5, 128, 128)
    pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    
    # flatten
    flat_conv, ftDim = _flatten(pool3)
    
    # FC-1 with 512 neurons
    fc_neurons = 3
    features = do_fc('fc_1', flat_conv, ftDim, fc_neurons, stdVal=0.05, wt_d=0.0005, isBias=True, isTrain=isTrain)
    # features = parametric_relu(fc1)	    

    return features, fc_neurons	

def inference(images, isTrain=True, toClassify=True):
    # Get features from basic CNN model
    features, ft_dim =compute_cnn_features(images, isTrain=isTrain, toClassify=toClassify) 
    
    # unit normalized
    features = tf.nn.l2_normalize(features,dim=1)
    kappa = tf.get_variable('kappa', dtype=tf.float32, initializer=tf.constant(1.22), trainable=True)
    features = tf.multiply(kappa, features, name='kappa_x')
  	
    if toClassify:
      # FC
      softmax_id = do_fc_norm('vMFML', features, ft_dim, NUM_CLASSES, wt_d=0.0005, biasVal=0.0, isTrain=isTrain)
      if(do_print):
        print(softmax_id)

      return softmax_id, features, kappa
    else:
      return features

def loss_L2():  
  l2Loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return l2Loss

def loss_softmax(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
          
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  logits=logits, labels=labels, name='cross_entropy_per_example')
  
  ent_loss = tf.reduce_mean(cross_entropy, name='entropy_loss')
  
  tf.add_to_collection('losses', ent_loss)
  
  return ent_loss

def loss_total():
  # The total loss is defined as the sum of all losses
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
