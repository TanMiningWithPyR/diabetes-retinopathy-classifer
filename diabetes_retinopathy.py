
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
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0       # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001     # Initial learning rate.

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


def _variable_with_weight_decay(name, shape, gain, wd, use_fp16):
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
      tf.orthogonal_initializer(gain=gain, seed=None, dtype=dtype),
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
                                              batch_size=flags.batch_size, repeat=flags.num_epochs)

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

 

def inference(left_images, right_images, left_indication, right_indication, flags):
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
                                         shape=[7, 7, 3, 32],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(input=left_images, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
    conv_R = tf.nn.conv2d(input=right_images, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases)
    conv_1L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_1R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_1L)
    _activation_summary(conv_1R)
    
  # pool1
  pool_1L = tf.nn.max_pool(conv_1L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1L')
  pool_1R = tf.nn.max_pool(conv_1R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1R')
  
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(pool_1L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(pool_1R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases)    
    conv_2L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_2R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_2L)
    _activation_summary(conv_2R)
    
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_2L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_2R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases)
    conv_3L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_3R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_3L)   
    _activation_summary(conv_3R)   
    
  # pool2
  pool_2L = tf.nn.max_pool(conv_3L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2L')
  pool_2R = tf.nn.max_pool(conv_3R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2R')
  
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 64],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(pool_2L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(pool_2R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases)        
    conv_4L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_4R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)    
    _activation_summary(conv_4L)
    _activation_summary(conv_4R)
    
  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_4L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_4R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_5L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_5R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_5L)   
    _activation_summary(conv_5R)   

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_5L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_5R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_6L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_6R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_6L)   
    _activation_summary(conv_6R)   
    
  # pool3
  pool_3L = tf.nn.max_pool(conv_6L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3L')  
  pool_3R = tf.nn.max_pool(conv_6R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3R') 
  # conv7
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(pool_3L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(pool_3R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_7L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_7R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)    
    _activation_summary(conv_7L)
    _activation_summary(conv_7R) 

  # conv8
  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_7L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_7R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_8L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_8R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_8L)   
    _activation_summary(conv_8R)   
    
  # conv9
  with tf.variable_scope('conv9') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_8L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_8R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_9L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_9R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_9L)    
    _activation_summary(conv_9R)      
    
  # conv10
  with tf.variable_scope('conv10') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_9L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_9R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_10L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_10R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_10L)   
    _activation_summary(conv_10R)   
    
  # pool4
  pool_4L = tf.nn.max_pool(conv_10L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4L')   
  pool_4R = tf.nn.max_pool(conv_10R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4R') 
  
  # conv11
  with tf.variable_scope('conv11') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(pool_4L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(pool_4R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_11L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_11R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_11L)
    _activation_summary(conv_11R)
    
  # conv12
  with tf.variable_scope('conv12') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_11L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_11R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_12L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_12R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_12L)   
    _activation_summary(conv_12R) 
    
  # conv13
  with tf.variable_scope('conv13') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_12L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_12R, kernel, [1, 1, 1, 1], padding='SAME')  
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_13L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_13R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)    
    _activation_summary(conv_13L)    
    _activation_summary(conv_13R) 
    
  # conv14
  with tf.variable_scope('conv14') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         gain=0.1,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv_L = tf.nn.conv2d(conv_13L, kernel, [1, 1, 1, 1], padding='SAME')
    conv_R = tf.nn.conv2d(conv_13R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), flags.use_fp16)
    pre_activation_L = tf.nn.bias_add(conv_L, biases)
    pre_activation_R = tf.nn.bias_add(conv_R, biases) 
    conv_14L = tf.nn.leaky_relu(pre_activation_L, alpha=0.5 ,name=scope.name)
    conv_14R = tf.nn.leaky_relu(pre_activation_R, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_14L)   
    _activation_summary(conv_14R)
    
  # pool5
  pool_5L = tf.nn.max_pool(conv_14L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5L') 
  pool_5R = tf.nn.max_pool(conv_14R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5R')   
  
  # dropout1
  dropout_1L = tf.nn.dropout(pool_5L, keep_prob=0.25, name='dropout1L')  
  dropout_1R = tf.nn.dropout(pool_5R, keep_prob=0.25, name='dropout1L')  
  
  # maxout1
  with tf.variable_scope('maxout1') as scope:
    reshape_L = tf.reshape(dropout_1L, [-1, 16384])
    reshape_R = tf.reshape(dropout_1R, [-1, 16384])
    weights = _variable_with_weight_decay('weights', shape=[16384, 512, 2],
                                          gain=0.1, wd=0.0, use_fp16=flags.use_fp16)    
    biases = _variable_on_cpu('biases', [512, 2], tf.constant_initializer(0.1), flags.use_fp16)
    maxout_L = tf.tensordot(reshape_L, weights, axes=1) + biases
    maxout_1L = tf.reduce_max(maxout_L, axis=2, name=scope.name)
    maxout_R = tf.tensordot(reshape_R, weights, axes=1) + biases
    maxout_1R = tf.reduce_max(maxout_R, axis=2, name=scope.name)
    _activation_summary(maxout_1L)
    _activation_summary(maxout_1R)
    
  # concat1
  with tf.variable_scope('concat1') as scope:
    concat_1L = tf.concat([maxout_1L,left_indication], axis=-1)
    concat_1R = tf.concat([maxout_1R,left_indication], axis=-1) 
      
  # merge two eyes
    
  # reshape1M(merge eyes)
  reshape_1M = tf.concat([concat_1L,concat_1R], axis=-1, name="reshape1M")

  # dropout2M
  dropout_2M = tf.nn.dropout(reshape_1M, keep_prob=0.25, name='dropout2M')    
  
  # maxout2
  with tf.variable_scope('maxout2') as scope:
    reshape = tf.reshape(dropout_2M, [-1, 1028])
    weights = _variable_with_weight_decay('weights', shape=[1028, 512, 2],
                                          gain=0.1, wd=0.0, use_fp16=flags.use_fp16)    
    biases = _variable_on_cpu('biases', [512, 2], tf.constant_initializer(0.1), flags.use_fp16)
    maxout = tf.tensordot(reshape, weights, axes=1) + biases
    maxout_2M = tf.reduce_max(maxout, axis=2, name='maxout2')
    _activation_summary(maxout_2M)  

  # dropout3M
  dropout_3M = tf.nn.dropout(maxout_2M, keep_prob=0.25, name='dropout3M')
    
  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, 2*NUM_CLASSES],
                                          gain=0.1, wd=0.0, use_fp16=flags.use_fp16)
    biases = _variable_on_cpu('biases', [2*NUM_CLASSES], tf.constant_initializer(0.1), flags.use_fp16)
    softmax_linear = tf.add(tf.matmul(dropout_3M, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
    
  # back to one eye
  reshape_2L, reshape_2R = tf.split(softmax_linear,[5,5],-1)
    
  return reshape_2L, reshape_2R  
  
def loss(left_logits, right_logits, left_labels, right_labels, examples_num):
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
  left_labels = tf.cast(left_labels, tf.int64)
  right_labels = tf.cast(right_labels, tf.int64)
  left_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=left_labels, logits=left_logits, name='left_cross_entropy_per_example')
  right_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=right_labels, logits=right_logits, name='right_cross_entropy_per_example')
  left_cross_entropy_mean = tf.reduce_mean(left_cross_entropy, name='left_cross_entropy')
  right_cross_entropy_mean = tf.reduce_mean(right_cross_entropy, name='right_cross_entropy')
  cross_entropy_mean = (left_cross_entropy_mean + right_cross_entropy_mean) / 2 
  # kappa loss
  left_labels_one_hot = tf.one_hot(left_labels, 5)
  right_labels_one_hot = tf.one_hot(right_labels, 5) 
  left_kappa_loss = quadratic_kappa.quadratic_kappa_on_one_hot_value(left_logits, left_labels_one_hot, 32)
  right_kappa_loss = quadratic_kappa.quadratic_kappa_on_one_hot_value(right_logits,right_labels_one_hot, 32)
  kappa_loss_mean = (left_kappa_loss + right_kappa_loss) / 2
  combined_loss = tf.clip_by_value(kappa_loss_mean, 0, 10 ** 3) + 0.5 * (tf.clip_by_value(cross_entropy_mean, 0, 10 ** 3))
  tf.add_to_collection('losses', combined_loss)
  return combined_loss
#---------------------------------------------------------------------
#  # The total loss is defined as the cross entropy loss plus all of the weight
#  # decay terms (L2 loss).
#  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in diabetes_retinopathy model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
#  # Compute the moving average of all individual losses and the total loss.
#  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
#  losses = tf.get_collection('losses')
#  loss_averages_op = loss_averages.apply(losses + [total_loss])
#
#  # Attach a scalar summary to all individual losses and the total loss; do the
#  # same for the averaged version of the losses.
#  for l in losses + [total_loss]:
#    # Name each loss as '(raw)' and name the moving average version of the loss
#    # as the original loss name.
#    tf.summary.scalar(l.op.name + ' (raw)', l)
#    tf.summary.scalar(l.op.name, loss_averages.average(l))
#
#  return loss_averages_op
  
#------------------------------------------------------------
  tf.summary.scalar('loss', total_loss)
  return total_loss

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

def eval_train_data(left_logits,right_logits,left_labels,right_labels):
  left_top_k_op = tf.nn.in_top_k(left_logits, left_labels, 1)
  right_top_k_op = tf.nn.in_top_k(right_logits, right_labels, 1)
  top_k_op = tf.concat([left_top_k_op, right_top_k_op], axis=-1)
  accuracy_op = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
  _activation_summary(accuracy_op)
  tf.add_to_collection('accuracy', accuracy_op)
  return tf.reduce_mean(tf.get_collection('accuracy'), name='total_accuracy') 

if __name__ == '__main__':
  FLAGS = model_parser.parse_args()