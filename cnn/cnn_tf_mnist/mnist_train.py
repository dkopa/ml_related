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
import mnist_vmfml_xavier as mncnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/raid5/hasnat/mnist_verify/trlogs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

INITIAL_LEARNING_RATE = 0.01
NUM_CLASSES = 10
FT_DIM = 3

imHeight = 28
imWidth = 28
numCh = 1

saveLogFile = True
isSaveModel = True

isModelFT = False
restoreFileName = "/raid5/hasnat/mnist_verify/trlogs/model-mn_vmfml_kappa_1_2.ckpt-9000"

netName = 'mn_vmfml_kappa_2_var'

if isModelFT:
    fname = 'train_log_'+ netName + '_ft_%s.txt'% (datetime.now())
else:
    fname = 'train_log_'+ netName + '_%s.txt'% (datetime.now())

def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    #### For training CNN
    images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, imHeight, imWidth, numCh])
    labels = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
    #####
    
    # Get images and labels
    images_tr, labels_tr = mnistip.distorted_inputs(randFlip=True)
    images_ev, labels_ev = mnistip.inputs(eval_data=True)

    # Build a Graph that computes the logits predictions
    # local, logits_id = mnist_net.inference_id_classification(images)
    logits_id, local, kappaVal = mncnn.inference(images)
    
    # Calculate losses...
    # loss_L2 = mnist_net.loss_L2()
    #loss_softmax_id = mnist_net.loss_softmax(logits_id, labels)
    #loss_combined = mnist_net.loss_combined_no_param()
    loss_softmax_id = mncnn.loss_softmax(logits_id, labels)
    loss_combined = mncnn.loss_total()
    
    train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_combined)
    
    # Create a saver.
    saver = tf.train.Saver()

    if(isModelFT):
      saver1 = tf.train.Saver(tf.trainable_variables())    

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    if(isModelFT):
      saver1.restore(sess, restoreFileName)
    
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    if(saveLogFile):
        if(os.path.isfile(fname)):
            os.remove(fname)
        f_handle = open(fname,'a')

    # for step in xrange(1):       
    lr_ = INITIAL_LEARNING_RATE    
    mulsteplr = [6, 8, 10]
    
    stEpoch = 1
    mulsteplr = np.array(mulsteplr)
    currEpoch = int(stEpoch * FLAGS.batch_size / mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    mulstepCnt = np.where(currEpoch < mulsteplr)[0][0]
    lr_ = lr_**(mulstepCnt+1)
    
    for step in xrange(stEpoch, FLAGS.max_steps):
      # Learning rate decrease policy
      currEpoch = int(step * FLAGS.batch_size / mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
      if(currEpoch >= mulsteplr[mulstepCnt]):
          print(str(currEpoch) + ':: Decreasing learning rate')
          lr_ = 0.8 * lr_
          mulstepCnt = mulstepCnt + 1
    
      _images, _labels = sess.run([images_tr, labels_tr])
      
      _, lossID, lossComb, kappaVal_ = sess.run([train_op, loss_softmax_id, loss_combined, kappaVal],
                                        feed_dict={images: _images, labels: _labels, learning_rate: lr_})

      if step % 100 == 0:
          format_str = ('%s: Step %d, LR=%.4f, ID-loss = %.2f, T-loss = %.2f, Kappa = %.2f\n')
          print (format_str % (datetime.now(), step, lr_, lossID, lossComb, kappaVal_))
          if(saveLogFile):
              f_handle.write(format_str % (datetime.now(), step, lr_, lossID, lossComb, kappaVal_))
      
      assert not np.isnan(lossComb), 'Model diverged with loss = NaN'
            
      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          if(isSaveModel):
              if isModelFT:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model-'+netName+'-ft.ckpt')    
              else:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model-'+netName+'.ckpt')    
              print('saving model ...')
              saver.save(sess, checkpoint_path, global_step=step)
      
          ########################### 
          # Evaluate test set
          ###########################
          numTestStep = int(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size)
          predictions_id = np.ndarray(shape=(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, mnistip.NUM_CLASSES), dtype = np.float64)
    
          ftVec = np.ndarray(shape=(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,FT_DIM), dtype = np.float64)
          tLabels = np.ndarray(shape=(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL), dtype = np.float64)
    
          # Evaluate
          print("====== Evaluating ID classification ========\n")
          for step_ev in xrange(numTestStep):
              _images, _labels = sess.run([images_ev, labels_ev])
                
              stIndx = step_ev*FLAGS.batch_size
              edIndx = (step_ev+1)*FLAGS.batch_size
                         
              _fts, tpred_id  = sess.run([local, logits_id], feed_dict={images: _images})
                
              predictions_id[stIndx:edIndx, :] = np.asarray(tpred_id)
    
              ftVec[stIndx:edIndx, :] = np.asarray(_fts)
              tLabels[stIndx:edIndx] = np.asarray(_labels)
            
          obs_labels = np.argmax(predictions_id, axis=1)
    
          #print(lab.dtype)
          sum_ = np.sum(tLabels==obs_labels)
          acc_id = sum_/float(mnistip.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
        
          print('================================')        
          format_str = ('%s: Step %d, ID_Acc = %.5f\n')
          np.savez(netName+'_'+str(step), X=ftVec, y=tLabels)
          print (format_str % (datetime.now(), step, acc_id))
              
          if(saveLogFile):
              f_handle.write('==================================================\n')
              f_handle.write(format_str % (datetime.now(), step, acc_id))
              f_handle.write('==================================================\n')
      
def main(argv=None):  # pylint: disable=unused-argument
  print(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
