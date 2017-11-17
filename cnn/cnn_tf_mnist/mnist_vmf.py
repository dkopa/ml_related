import tensorflow as tf
import mnist_input

FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = mnist_input.NUM_CLASSES
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
                         
def inference(images, isTrain=True, toClassify=True, isBN=True, isUnitNorm = False):
  # conv1-pool1
  conv1 = do_conv_act('conv1', images, 5, 1, 32, stdVal=5e-2)
  pool1 = do_maxpool('pool1', conv1, 2, 2)
  
  # conv2-pool2
  conv2 = do_conv_act('conv2', pool1, 5, 32, 64, stdVal=5e-2)
  pool2 = do_maxpool('pool2', conv2, 2, 2)

  flat_pool, t_num_dim = _flatten(pool2)
                       
  # fc3 - activation
  fc3 = do_fc('local3', flat_pool, t_num_dim, 512, stdVal=0.04, wt_d=0.004, biasVal=0.1)
  fc3 = tf.nn.relu(fc3, name='local3')
    
  # fc4
  fc4 = do_fc('local4', fc3, 512, 3, stdVal=0.04, wt_d=0.004, biasVal=0.1, isBN=False)
  
  # unit normalized
  if(isUnitNorm):
      normVal = tf.sqrt(tf.reduce_sum(tf.square(fc4), 1, keep_dims=True))
      fc4 = fc4 / normVal
    
  if(do_print):
      print(fc4)
  
  if toClassify:
      with tf.variable_scope('softmax_linear') as scope:
        inpNumNeurons = 3
        weights = _variable_with_weight_decay('weights', [inpNumNeurons, NUM_CLASSES],stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc4, weights), biases, name=scope.name) 
        if(do_print):
            print(softmax_linear)
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

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  # return tf.add_n(tf.get_collection('losses'), name='total_loss')
  return cross_entropy_mean

def loss_vmf(local, labels):
    # Compute unit-normalized mean
    _, labels_ = tf.unique(labels)

    cntSums = tf.segment_mean(local, labels_, name='sm1')
    normVal = tf.sqrt(tf.reduce_sum(tf.square(cntSums), 1, keep_dims=True))
    cntSums = cntSums / normVal
    
    # Compute cosine-distance among features and mean
    diff_cents = tf.exp(tf.matmul(local, cntSums, transpose_b=True, name='mm1'))
    
    # Compute class probabilities
    sum_of_dists = tf.reduce_sum(diff_cents, keep_dims=True, reduction_indices=1)    
    prob_class = diff_cents / sum_of_dists
    
    # Compute cross entropy loss
    smloss = tf.nn.sparse_softmax_cross_entropy_with_logits(diff_cents, labels_)
  
    loss_mean = tf.reduce_mean(smloss, name='mean_loss')
    tf.add_to_collection('losses', loss_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).  
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss_mean

def loss_vmf_plus(local, labels):
    # Compute unit-normalized mean
    _, labels_ = tf.unique(labels)

    cntSums = tf.segment_mean(local, labels_, name='sm1')
    normVal = tf.sqrt(tf.reduce_sum(tf.square(cntSums), 1, keep_dims=True))
    cntSums = cntSums / normVal
    
    # Compute inter-cluster similarities as distance-loss
    simIntCl = tf.reduce_sum(tf.matmul(cntSums, cntSums, transpose_b=True, name='mm_int_clust') + 1) 
    simIntCl = simIntCl - (2 * (tf.cast(tf.reduce_max(labels_), tf.float32)+1))
    # tf.add_to_collection('losses', simIntCl)
    
    # Compute cosine-distance among features and mean
    diff_cents = tf.exp(tf.matmul(local, cntSums, transpose_b=True, name='mm1'))
    
    # Compute class probabilities
    sum_of_dists = tf.reduce_sum(diff_cents, keep_dims=True, reduction_indices=1)    
    prob_class = diff_cents / sum_of_dists
    
    # Compute cross entropy loss
    smloss = tf.nn.sparse_softmax_cross_entropy_with_logits(diff_cents, labels_)
  
    loss_mean = tf.reduce_mean(smloss, name='mean_loss')
    tf.add_to_collection('losses', loss_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).  
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss_mean, simIntCl, cntSums

def loss_vmf_plus_cent_update(local, labels, fc):
    # Update center
    cntrs = fc.get_embed_centers(local, labels)
    
    # Compute unit-normalized mean
    unq, labels_ = tf.unique(labels)

    #cntSums = tf.segment_mean(local, labels_, name='sm1')
    #normVal = tf.sqrt(tf.reduce_sum(tf.square(cntSums), 1, keep_dims=True))
    #cntSums = cntSums / normVal
    
    # Compute unit-normalized mean    
    cnts = tf.gather(cntrs, unq)
    
    # Compute inter-cluster similarities as distance-loss
    simIntCl = tf.reduce_sum(tf.matmul(cnts, cnts, transpose_b=True, name='mm_int_clust') + 1) 
    simIntCl = simIntCl - (2 * (tf.cast(tf.reduce_max(labels_), tf.float32)+1))
    # tf.add_to_collection('losses', simIntCl)
    
    # Compute cosine-distance among features and mean
    diff_cents = tf.exp(tf.matmul(local, cnts, transpose_b=True, name='mm1'))
    
    # Compute class probabilities
    #sum_of_dists = tf.reduce_sum(diff_cents, keep_dims=True, reduction_indices=1)    
    #prob_class = diff_cents / sum_of_dists
    
    # Compute cross entropy loss
    smloss = tf.nn.sparse_softmax_cross_entropy_with_logits(diff_cents, labels_)
  
    loss_mean = tf.reduce_mean(smloss, name='mean_loss')
    tf.add_to_collection('losses', loss_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).  
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss_mean, simIntCl, cnts
