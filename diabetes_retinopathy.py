
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

model_parser = argparse.ArgumentParser(parents=[diabetes_retinopathy_input.input_parser], add_help=False, 
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Basic model parameters.
model_parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

# Global constants describing the diabetes_retinopathy data set.
NUM_CLASSES = diabetes_retinopathy_input.NUM_CLASSES

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001    # Initial learning rate.

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
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


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


def _variable_with_weight_decay(name, shape, stddev, wd, use_fp16):
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
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
      use_fp16)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(data_dir, labels_file, num_epochs, batch_size, use_fp16):
  """Construct distorted input for diabetes_retinopathy training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir, If no labels_file
  """
  if not data_dir:
    raise ValueError('Please supply a data_dir')
  train_dir = os.path.join(data_dir,'train')
  if not os.path.exists(train_dir):
    raise ValueError('Please supply train folder in data_dir')
  if not labels_file:
    raise ValueError('Please supply a labels_file')  
  left_images,left_labels,right_images,right_labels,patients = \
  diabetes_retinopathy_input.distorted_inputs(data_dir=train_dir, labels_file=labels_file,
                                              num_epochs=num_epochs, batch_size=batch_size)
  if use_fp16:
    left_images = tf.cast(left_images, tf.float16)
    left_labels = tf.cast(left_labels, tf.float16)
    right_images = tf.cast(right_images, tf.float16)
    right_labels = tf.cast(right_labels, tf.float16)
    patients = tf.cast(patients, tf.float16)
  return left_images,left_labels,right_images,right_labels,patients


def inputs(eval_data, data_dir, labels_file, batch_size, use_fp16):
  """Construct input for diabetes_retinopathy evaluation using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir, If no labels_file
  """
  if not data_dir:
    raise ValueError('Please supply a data_dir')
  eval_dir = os.path.join(data_dir,eval_data)
  if not os.path.exists(eval_dir):
    raise ValueError('Please supply evaluation folder in data_dir')
  if not labels_file:
    raise ValueError('Please supply a labels_file') 
  left_images,left_labels,right_images,right_labels,patients = \
  diabetes_retinopathy_input.inputs(data_dir=eval_dir, labels_file=labels_file, batch_size=batch_size)
  if use_fp16:
    left_images = tf.cast(left_images, tf.float16)
    left_labels = tf.cast(left_labels, tf.float16)
    right_images = tf.cast(right_images, tf.float16)
    right_labels = tf.cast(right_labels, tf.float16)
    patients = tf.cast(patients, tf.float16)
  return left_images,left_labels,right_images,right_labels,patients

 

def inference(left_images, right_images, batch_size, use_fp16):
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
  
  # Left eyes layers
  # conv1L
  with tf.variable_scope('conv1L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 3, 32],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(input=left_images, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv1L)

  # pool1L
  pool1L = tf.nn.max_pool(conv1L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1L')

  # conv2L
  with tf.variable_scope('conv2L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool1L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv2L)

  # conv3L
  with tf.variable_scope('conv3L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv2L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv3L)   
    
  # pool2L
  pool2L = tf.nn.max_pool(conv3L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2L')
  
  # conv4L
  with tf.variable_scope('conv4L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 64],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool2L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv4L)

  # conv5L
  with tf.variable_scope('conv5L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv4L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv5L)   
    
  # pool3L
  pool3L = tf.nn.max_pool(conv5L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3L')  

  # conv6L
  with tf.variable_scope('conv6L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool3L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv6L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv6L)

  # conv7L
  with tf.variable_scope('conv7L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv6L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv7L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv7L)   

  # conv8L
  with tf.variable_scope('conv8L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv7L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv8L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv8L)   

  # conv9L
  with tf.variable_scope('conv9L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv8L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv9L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv9L)    
    
  # pool4L
  pool4L = tf.nn.max_pool(conv9L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4L')   

  # conv10L
  with tf.variable_scope('conv10L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool4L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv10L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv10L)

  # conv11L
  with tf.variable_scope('conv11L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv10L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv11L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv11L)   

  # conv12L
  with tf.variable_scope('conv12L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv11L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv12L)   

  # conv13L
  with tf.variable_scope('conv13L') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv12L, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13L = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv13L)    
    
  # pool5L
  pool5L = tf.nn.max_pool(conv13L, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5L') 
  
  # dropout1L
  dropout1L = tf.nn.dropout(pool5L, keep_prob=0.25, name='dropout1L')  
 
  # maxout1L
  with tf.variable_scope('maxout1L') as scope:
    reshape = tf.reshape(dropout1L, [-1, 16384])
    weights = _variable_with_weight_decay('weights', shape=[16384, 512, 2],
                                          stddev=0.04, wd=0.004, use_fp16=use_fp16)    
    biases = _variable_on_cpu('biases', [512, 2], tf.constant_initializer(0.1), use_fp16)
    maxout1L = tf.tensordot(reshape, weights, axes=1) + biases
    maxout1L = tf.reduce_max(maxout1L, axis=2, name='maxout1L')
    _activation_summary(maxout1L)
  
  # concat1L
  with tf.variable_scope('concat1L') as scope:
    concat1L = tf.concat([maxout1L,tf.constant(batch_size * [[1.0,0.0]])], axis=-1)


  # Right eyes layers
  # conv1R
  with tf.variable_scope('conv1R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 3, 32],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(input=right_images, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv1R)

  # pool1R
  pool1R = tf.nn.max_pool(conv1R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1R')

  # conv2R
  with tf.variable_scope('conv2R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool1R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv2R)

  # conv3R
  with tf.variable_scope('conv3R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv2R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv3R)   
    
  # pool2L
  pool2R = tf.nn.max_pool(conv3R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2R')
  
  # conv4R
  with tf.variable_scope('conv4R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 64],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool2R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv4R)

  # conv5R
  with tf.variable_scope('conv5R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv4R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv5R)   
    
  # pool3R
  pool3R = tf.nn.max_pool(conv5R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3R')  

  # conv6R
  with tf.variable_scope('conv6R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool3R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv6R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv6R)

  # conv7R
  with tf.variable_scope('conv7R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv6R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv7R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv7R)   

  # conv8R
  with tf.variable_scope('conv8R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv7R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv8R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv8R)   

  # conv9R
  with tf.variable_scope('conv9R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv8R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv9R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv9R)    
    
  # pool4R
  pool4R = tf.nn.max_pool(conv9R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4R')   

  # conv10R
  with tf.variable_scope('conv10R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(pool4R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv10R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv10R)

  # conv11R
  with tf.variable_scope('conv11R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv10R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv11R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv11R)   

  # conv12R
  with tf.variable_scope('conv12R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv11R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv12R)   

  # conv13R
  with tf.variable_scope('conv13R') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0,
                                         use_fp16=use_fp16)
    conv = tf.nn.conv2d(conv12R, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1), use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13R = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv13R)    
    
  # pool5R
  pool5R = tf.nn.max_pool(conv13R, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5R') 
  
  # dropout1R
  dropout1R = tf.nn.dropout(pool5R, keep_prob=0.25, name='dropout1R')  
 
  # maxout1R
  with tf.variable_scope('maxout1R') as scope:
    reshape = tf.reshape(dropout1R, [-1, 16384])
    weights = _variable_with_weight_decay('weights', shape=[16384, 512, 2],
                                          stddev=0.04, wd=0.004, use_fp16=use_fp16)    
    biases = _variable_on_cpu('biases', [512, 2], tf.constant_initializer(0.1), use_fp16)
    maxout1R = tf.tensordot(reshape, weights, axes=1) + biases
    maxout1R = tf.reduce_max(maxout1R, axis=2, name='maxout1R')
    _activation_summary(maxout1R)
  
  # concat1R
  with tf.variable_scope('concat1R') as scope:
    concat1R = tf.concat([maxout1R,tf.constant(batch_size * [[0.0,1.0]])], axis=-1)
      
  # merge two eyes
    
  # reshape1M(merge eyes)
  reshape1M = tf.concat([concat1L,concat1R], axis=-1)

  # dropout2M
  dropout2M = tf.nn.dropout(reshape1M, keep_prob=0.25, name='dropout2M')    
  
  # maxout2
  with tf.variable_scope('maxout2') as scope:
    reshape = tf.reshape(dropout2M, [-1, 1028])
    weights = _variable_with_weight_decay('weights', shape=[1028, 512, 2],
                                          stddev=0.04, wd=0.004, use_fp16=use_fp16)    
    biases = _variable_on_cpu('biases', [512, 2], tf.constant_initializer(0.1), use_fp16)
    maxout2 = tf.tensordot(reshape, weights, axes=1) + biases
    maxout2 = tf.reduce_max(maxout2, axis=2, name='maxout2')
    _activation_summary(maxout2)  

  # dropout3M
  dropout3M = tf.nn.dropout(maxout2, keep_prob=0.25, name='dropout3M')
    
  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, 2*NUM_CLASSES],
                                          stddev=1/512.0, wd=0.0, use_fp16=use_fp16)
    biases = _variable_on_cpu('biases', [2*NUM_CLASSES], tf.constant_initializer(0.0), use_fp16)
    softmax_linear = tf.add(tf.matmul(dropout3M, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
    
  # back to one eye
  reshape2L, reshape2R = tf.split(softmax_linear,[5,5],-1)
    
  return reshape2L, reshape2R  
  
def loss(left_logits, right_logits, left_labels, right_labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  left_labels = tf.cast(left_labels, tf.int64)
  right_labels = tf.cast(right_labels, tf.int64)
  left_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=left_labels, logits=left_logits, name='left_cross_entropy_per_example')
  right_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=right_labels, logits=right_logits, name='right_cross_entropy_per_example')
  left_cross_entropy_mean = tf.reduce_mean(left_cross_entropy, name='left_cross_entropy')
  right_cross_entropy_mean = tf.reduce_mean(right_cross_entropy, name='right_cross_entropy')
  cross_entropy_mean = (left_cross_entropy_mean + right_cross_entropy_mean) / 2 
  tf.add_to_collection('losses', cross_entropy_mean)

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


def train(total_loss, global_step, batch_size, num_examples_for_train):
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
  num_batches_per_epoch = num_examples_for_train / batch_size
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
    opt = tf.train.GradientDescentOptimizer(lr)
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

if __name__ == '__main__':
  FLAGS = model_parser.parse_args()