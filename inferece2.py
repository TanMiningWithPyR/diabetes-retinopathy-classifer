# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:58:38 2018

@author: admin
"""

import tensorflow as tf

def _activation_summary(x):
  pass

def _variable_on_cpu(name, shape, initializer, use_fp16):
  pass 

def _variable_with_weight_decay(name, shape, init_parameter, wd, use_fp16):
  pass

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
                                         shape=[7, 7, 3, 32],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(input=images, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_1 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_1)

    
  # pool1
  pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')  
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases) 
    conv_2 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_2)
    
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_3 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_3)   
  
    
  # pool2
  pool_2 = tf.nn.max_pool(conv_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')
  
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 64],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)    
    conv_4 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_4)
    
  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_5 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_5)  

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_6 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_6)   
    
  # pool3
  pool_3 = tf.nn.max_pool(conv_6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')  
  # conv7
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_3, kernel, [1, 1, 1, 1], padding='SAME')

    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_7 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)   
    _activation_summary(conv_7)

  # conv8
  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_7, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases) 
    conv_8 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_8)   
    
  # conv9
  with tf.variable_scope('conv9') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_8, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_9 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_9)       
    
  # conv10
  with tf.variable_scope('conv10') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_9, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_10 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_10)   
    
  # pool4
  pool_4 = tf.nn.max_pool(conv_10, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')     
  # conv11
  with tf.variable_scope('conv11') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(pool_4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_11 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_11)
    
  # conv12
  with tf.variable_scope('conv12') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_11, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_12 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name) 
    _activation_summary(conv_12)   
    
  # conv13
  with tf.variable_scope('conv13') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_12, kernel, [1, 1, 1, 1], padding='SAME')  
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_13 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)   
    _activation_summary(conv_13)    
    
  # conv14
  with tf.variable_scope('conv14') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         init_parameter=0.001,
                                         wd=0.0,
                                         use_fp16=flags.use_fp16)
    conv = tf.nn.conv2d(conv_13, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), flags.use_fp16)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_14 = tf.nn.leaky_relu(pre_activation, alpha=0.5 ,name=scope.name)
    _activation_summary(conv_14)   

  # pool5
  pool_5 = tf.nn.max_pool(conv_14, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5')     
  # dropout1
  dropout_1 = tf.nn.dropout(pool_5, keep_prob=0.25, name='dropout1')   
  
  # maxout1
  with tf.variable_scope('maxout1') as scope:
    # reshape from convolution
    reshape = tf.reshape(dropout_1, [-1, 16384])
    weights = _variable_with_weight_decay('weights', shape=[16384, 512, 2],
                                          init_parameter=0.001, wd=0.0, use_fp16=flags.use_fp16)    
    biases = _variable_on_cpu('biases', [512, 2], tf.constant_initializer(0.0), flags.use_fp16)
    maxout = tf.tensordot(reshape, weights, axes=1) + biases
    maxout_1 = tf.reduce_max(maxout, axis=2, name=scope.name)
    _activation_summary(maxout_1)   
      
  # merge two eyes
  with tf.variable_scope('reshape1') as scope:
    # concat left and right lable
    concat_1 = tf.concat([maxout_1,indication], axis=-1)
    # reshape1(merge eyes)
    reshape_1 = tf.reshape(concat_1, [32,-1], name=scope.name)
    
  # dropout2
  dropout_2 = tf.nn.dropout(reshape_1, keep_prob=0.25, name='dropout2')    
  
  # maxout2
  with tf.variable_scope('maxout2') as scope:
    reshape = tf.reshape(dropout_2, [-1, 1028])
    weights = _variable_with_weight_decay('weights', shape=[1028, 512, 2],
                                          init_parameter=0.001, wd=0.0, use_fp16=flags.use_fp16)    
    biases = _variable_on_cpu('biases', [512, 2], tf.constant_initializer(0.0), flags.use_fp16)
    maxout = tf.tensordot(reshape, weights, axes=1) + biases
    maxout_2 = tf.reduce_max(maxout, axis=2, name='maxout2')
    _activation_summary(maxout_2)  

  # dropout3
  dropout_3 = tf.nn.dropout(maxout_2, keep_prob=0.25, name='dropout3')
    
  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, 2*NUM_CLASSES],
                                          init_parameter=0.001, wd=0.0, use_fp16=flags.use_fp16)
    biases = _variable_on_cpu('biases', [2*NUM_CLASSES], tf.constant_initializer(0.0), flags.use_fp16)
    softmax_linear = tf.add(tf.matmul(dropout_3, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
    
  # back to one eye
  with tf.variable_scope('reshape2') as scope:
    reshape_2 = tf.reshape(softmax_linear,[64,5],name=scope.name)
    
  return reshape_2