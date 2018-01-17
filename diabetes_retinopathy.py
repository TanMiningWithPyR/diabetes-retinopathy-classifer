
"""Builds the diabetes_retinopathy network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re

import tensorflow as tf
import numpy as np

import diabetes_retinopathy_input
import quadratic_kappa

model_parser = argparse.ArgumentParser(parents=[diabetes_retinopathy_input.input_parser], add_help=False, 
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Basic model parameters.
model_parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

# Global constants describing the diabetes_retinopathy data set.
NUM_CLASSES = diabetes_retinopathy_input.NUM_CLASSES

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999               # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0                 # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = np.sqrt(0.1)   # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001               # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, use_fp16):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, init_parameter, wd, use_fp16):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
#      tf.truncated_normal_initializer(stddev=init_parameter, dtype=dtype),
#      tf.orthogonal_initializer(gain=init_parameter, seed=None, dtype=dtype),
      None,
      use_fp16)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(flags):
  """Construct distorted input for diabetes_retinopathy training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir, If no labels_file
  """
  if not flags.data_dir:
    raise ValueError('Please supply a data_dir')
  train_dir = os.path.join(flags.data_dir,'train')
  if not os.path.exists(train_dir):
    raise ValueError('Please supply train folder in data_dir')
  if not flags.labels_file:
    raise ValueError('Please supply a labels_file')  
  left_images,left_labels,right_images,right_labels,patients,examples_num,left_indication,right_indication = \
  diabetes_retinopathy_input.distorted_inputs(data_dir=train_dir, labels_file=flags.labels_file,
                                              batch_size=flags.batch_size, 
                                              imbalance_ratio=flags.imbalance_ratio,
                                              repeat=flags.num_epochs)

  if flags.use_fp16:
    left_images = tf.cast(left_images, tf.float16)
    left_labels = tf.cast(left_labels, tf.float16)
    right_images = tf.cast(right_images, tf.float16)
    right_labels = tf.cast(right_labels, tf.float16)
    patients = tf.cast(patients, tf.float16)
    left_indication = tf.cast(left_indication, tf.float16)
    right_indication = tf.cast(right_indication, tf.float16)
  return left_images,left_labels,right_images,right_labels,patients,examples_num,left_indication,right_indication


def inputs(flags):
  """Construct input for diabetes_retinopathy evaluation using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir, If no labels_file
  """
  if not flags.data_dir:
    raise ValueError('Please supply a data_dir')
  eval_dir = os.path.join(flags.data_dir,flags.eval_data)
  if not os.path.exists(eval_dir):
    raise ValueError('Please supply evaluation folder in data_dir')
  if not flags.labels_file:
    raise ValueError('Please supply a labels_file') 
  left_images,left_labels,right_images,right_labels,patients,left_indication,right_indication = \
  diabetes_retinopathy_input.inputs(data_dir=eval_dir, labels_file=flags.labels_file, batch_size=flags.batch_size)

  if flags.use_fp16:
    left_images = tf.cast(left_images, tf.float16)
    left_labels = tf.cast(left_labels, tf.float16)
    right_images = tf.cast(right_images, tf.float16)
    right_labels = tf.cast(right_labels, tf.float16)
    patients = tf.cast(patients, tf.float16)
    left_indication = tf.cast(left_indication, tf.float16)
    right_indication = tf.cast(right_indication, tf.float16)
  return left_images,left_labels,right_images,right_labels,patients,left_indication,right_indication

def inference(images, indication, flags):
  """Build the diabetes retinopathy model.

  Args:
    left_images, right_images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().  
  
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 32],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(input=images, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_1 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)  
    conv_2 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_2)
    
  # pool1
  pool_1 = tf.nn.max_pool(conv_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 32, 64],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_3 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_3)
    
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)      
    conv_4 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_4)

    
  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_5 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_5)   
 
  # pool2
  pool_2 = tf.nn.max_pool(conv_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_6 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_6)        

  # conv7
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_6, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_7 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_7)

  # conv8
  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_7, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_8 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_8)     

  # pool3
  pool_3 = tf.nn.max_pool(conv_8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')  
  
  # conv9
  with tf.variable_scope('conv9') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_9 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_9)        
    
  # conv10
  with tf.variable_scope('conv10') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_9, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_10 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_10)      
  
  # conv11
  with tf.variable_scope('conv11') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_10, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_11 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_11)
    
  # pool4
  pool_4 = tf.nn.max_pool(conv_11, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')   
  
  # conv12
  with tf.variable_scope('conv12') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 512],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_12 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_12)   
    
  # conv13
  with tf.variable_scope('conv13') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 512, 512],
                                         init_parameter=1.0,
                                         wd=5e-4,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_12, kernel, [1, 1, 1, 1], padding='SAME') 
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_13 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)  
    _activation_summary(conv_13)    
    
  # l2-pool5
  pool_5 = tf.sqrt(tf.nn.avg_pool(tf.square(conv_13), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME'), name='pool5')  

  # dropout1
  dropout_1 = tf.nn.dropout(pool_5, keep_prob=0.25, name='dropout1')   
  
#  # merge two eyes
#  with tf.variable_scope('reshape1') as scope:
#    # reshape from convolution
#    reshape = tf.reshape(dropout_1, [-1, 4*4*512])
#    # concat left and right lable
#    concat = tf.concat([reshape, indication], axis=-1)
#    # reshape1(merge eyes)
#    reshape_1 = tf.reshape(concat, [32,-1], name=scope.name)
#
#  # Dense1
#  with tf.variable_scope('Dense1') as scope:
#    weights = _variable_with_weight_decay('weights', shape=[4*4*512*2+4, 1024],
#                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)    
#    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0), flags.use_fp16)
#    pre_activation = tf.matmul(reshape_1, weights) + biases
#    Dense_1 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
#    _activation_summary(Dense_1)
#      
#  # Dense2
#  with tf.variable_scope('Dense2') as scope:
#    weights = _variable_with_weight_decay('weights', shape=[2048, 1024],
#                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)    
#    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0), flags.use_fp16)
#    pre_activation = tf.matmul(Dense_1, weights) + biases
#    Dense_2 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
#    _activation_summary(Dense_2)
#  
#  # Dense3
#  with tf.variable_scope('Dense3') as scope:
#    weights = _variable_with_weight_decay('weights', shape=[1024, 512],
#                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)    
#    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), flags.use_fp16)
#    pre_activation = tf.matmul(Dense_2, weights) + biases
#    Dense_3 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
#    _activation_summary(Dense_3)  
#
#  # dropout2
#  dropout_2 = tf.nn.dropout(Dense_3, keep_prob=0.25, name='dropout2')      
#    
#  # linear layer(WX + b),
#  # We don't apply softmax here because
#  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
#  # and performs the softmax internally for efficiency.
#  with tf.variable_scope('softmax_linear') as scope:
#    weights = _variable_with_weight_decay('weights', [512, 2*NUM_CLASSES],
#                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)
#    biases = _variable_on_cpu('biases', [2*NUM_CLASSES], tf.constant_initializer(0.0), flags.use_fp16)
#    softmax_linear = tf.add(tf.matmul(dropout_2, weights), biases, name=scope.name)
#    _activation_summary(softmax_linear)
#    
#  # back to one eye
#  with tf.variable_scope('reshape2') as scope:
#    reshape_2 = tf.reshape(softmax_linear,[64,5],name=scope.name)

#  return reshape_2
########################################################################################################  
  # Don't merge, compare with keras  
  with tf.variable_scope('reshape1') as scope:
    # reshape from convolution
    reshape1 = tf.reshape(dropout_1, [-1, 4*4*512])
    
  # Dense1
  with tf.variable_scope('Dense1') as scope:
    weights = _variable_with_weight_decay('weights', shape=[4*4*512, 1024],
                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)    
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.matmul(reshape1, weights) + biases
    Dense_1 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(Dense_1)
      
  # Dense2
  with tf.variable_scope('Dense2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 1024],
                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)    
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.matmul(Dense_1, weights) + biases
    Dense_2 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(Dense_2)
  
  # Dense3
  with tf.variable_scope('Dense3') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)    
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.matmul(Dense_2, weights) + biases
    Dense_3 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(Dense_3)  

  # dropout2
  dropout_2 = tf.nn.dropout(Dense_3, keep_prob=0.25, name='dropout2')      
    
  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES],
                                          init_parameter=1.0, wd=5e-4, use_fp16=flags.use_fp16)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0), flags.use_fp16)
    softmax_linear = tf.add(tf.matmul(dropout_2, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
#########################################################################################################
  return softmax_linear


  
def loss(logits, labels, examples_num):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [samples_count]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  # the sparse_softmax_cross_entropy_with_logits is support one-hot labels, this is important
  labels = tf.cast(labels, tf.int64)  
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='left_cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='left_cross_entropy')

  # kappa loss
#  labels_one_hot = tf.one_hot(labels, 5)
#  kappa_loss = quadratic_kappa.quadratic_kappa_on_one_hot_value(tf.nn.softmax(logits), labels_one_hot, 32)
#  combined_loss = kappa_loss_mean + 1 * (tf.clip_by_value(cross_entropy_mean, 0, 10 ** 3))
  combined_loss = cross_entropy_mean
  tf.add_to_collection('losses', combined_loss)
#---------------------------------------------------------------------
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in diabetes_retinopathy model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
  
#------------------------------------------------------------
#  tf.summary.scalar('loss', total_loss)
#  return total_loss

def train(total_loss, global_step, flags):
  """Train diabetes_retinopathy model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = flags.num_examples_for_train / flags.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def eval_train_data(logits,labels):
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  accuracy_op = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
  _activation_summary(accuracy_op)
  tf.add_to_collection('accuracy', accuracy_op)
  return tf.reduce_mean(tf.get_collection('accuracy'), name='total_accuracy') 

if __name__ == '__main__':
  FLAGS = model_parser.parse_args()