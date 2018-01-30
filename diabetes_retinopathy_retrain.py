# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:43:55 2018

@author: admin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import math
import argparse
import tensorflow as tf
import numpy as np

import diabetes_retinopathy

train_parser = argparse.ArgumentParser(parents=[diabetes_retinopathy.model_parser], 
                                       add_help=True, 
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

train_parser.add_argument('--train_log_dir', type=str, default='D:\\AlanTan\\CNN\\diabetes_retinopathy_tensorflow\\diabetes_retinopathy_classifer_tensorflow_restore_train',
                    help='Directory where to write event logs and checkpoint.')

train_parser.add_argument('--checkpoint_dir', type=str, default='D:\\AlanTan\\CNN\\diabetes_retinopathy_tensorflow\\diabetes_retinopathy_classifer_tensorflow_train2',
                    help='Directory where to read model checkpoints.')

train_parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

train_parser.add_argument('--log_frequency', type=int, default=-1,
                    help='How often to log results to the console.-1 is one epoch')

def re_train(last_step):
  """Train diabete_retinopathy for a number of steps."""

  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    # Get images and labels for diabete_retinopathy.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      left_images, left_labels, right_images, right_labels, patients, examples_num, left_indication, right_indication = \
      diabetes_retinopathy.distorted_inputs(FLAGS, global_step)
    
    labels = tf.concat([left_labels, right_labels], 0)
    images = tf.concat([left_images, right_images], 0)
    indication = tf.concat([left_indication, right_indication], 0)
    
    # Build a Graph that computes the logits predictions from the
    # inference model.    
    logits = diabetes_retinopathy.inference(images, indication, FLAGS)

    # Calculate loss.
    loss = diabetes_retinopathy.loss(logits, labels, examples_num)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = diabetes_retinopathy.train(loss, global_step, FLAGS)  
    
    # Calculate eval accuracy.
    with tf.device('/cpu:0'):
      e_left_images, e_left_labels, e_right_images, e_right_labels, e_patients, e_left_indication, e_right_indication = \
      diabetes_retinopathy.inputs(FLAGS)   
     
    e_labels = tf.concat([e_left_labels, e_right_labels], 0)
    e_images = tf.concat([e_left_images, e_right_images], 0)   
    e_indication = tf.concat([e_left_indication, e_right_indication], 0)
    e_logits = diabetes_retinopathy.inference(e_images, e_indication, FLAGS)
    
#    accuracy_op_eval = diabetes_retinopathy.eval_train_data(e_logits, e_labels)
    
    pre_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    
    class _RunValue():
      def __init__(self):
        self.results = [5.0, 0.0] 

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""
      def __init__(self):
        super(_LoggerHook, self).__init__()
        if FLAGS.log_frequency == -1:   # how many steps in one epoch 
          self._log_frequency = int(math.ceil(FLAGS.num_examples_for_train / FLAGS.batch_size))
        else:
          self.log_frequency = FLAGS.log_frequency  
          
      def begin(self):
        self._step = -1    
      
      def before_run(self, run_context):
        self._step += 1          
        if self._step % self._log_frequency == 0:  
          self._start_time = time.time()
          self._epoch = self._step // self._log_frequency
          format_str = ('%s: epoch %d start (%d totally)')
          print (format_str % (datetime.now(), self._epoch, FLAGS.num_epochs))          
        return tf.train.SessionRunArgs([loss])  # Asks for loss.

      def after_run(self, run_context, run_values):          
        if self._step % self._log_frequency == self._log_frequency - 1 :
          current_time = time.time()
          duration = current_time - self._start_time

          loss_value = run_values.results[0]
          steps_per_sec = self._log_frequency / duration
          sec_per_sec = float(duration / self._log_frequency)
          format_str = ('%s: step %d, epoch %d end, loss = %.3f (%.3f steps/sec; %.3f sec/step)')
          print (format_str % (datetime.now(), self._step, self._epoch, loss_value,
                               steps_per_sec, sec_per_sec))           

    class _saverHook(tf.train.SessionRunHook):
      def __init__(self, checkpoint_dir, save_steps, saver, checkpoint_basename="model.ckpt",):
        super(_saverHook, self).__init__()
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._save_steps = save_steps
        
      def begin(self):
        self._step = -1
        self._saved_accurancy = 0.0
        self._accurancy_index = 0
    
      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([loss])
  
      def after_run(self, run_context, run_values):
        if self._step % self._save_steps == self._save_steps - 1 or self._step == 0:   
#          self._accuracy_on_eval = run_context.session.run(accuracy_op_eval)
          self._accuracy_on_eval = diabetes_retinopathy.eval_train_data(run_context.session, e_logits, e_labels, FLAGS)
          if self._accuracy_on_eval >= self._saved_accurancy:              
            self._saved_accurancy = self._accuracy_on_eval
            self._saver.save(run_context.session, self._save_path, self._step) 
            train_format_str = ('%s: step %d, loss = %.3f')
            eval_format_str = ('%s: step %d, eval_accuracy_value = %.3f, weights was saved')
            print (train_format_str % (datetime.now(), self._step, run_values.results[0])) 
            print (eval_format_str % (datetime.now(), self._step, self._saved_accurancy))   
          else:
            eval_format_str = ('%s: step %d, eval_accuracy_value = %.3f, weights was not saved')
            print (eval_format_str % (datetime.now(), self._step, self._accuracy_on_eval))  
                
        if self._step % 20 == 0:
          format_str = ('%s: step %d, loss = %.5f')
          print (format_str % (datetime.now(), self._step, run_values.results[0]))             
          
    mysaver=tf.train.Saver(max_to_keep=6)
    
    with tf.Session() as restore_sess:
      tf.global_variables_initializer().run()
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
          # Restores from checkpoint
        pre_saver.restore(restore_sess, ckpt.model_checkpoint_path)
          # Assuming model_checkpoint_path looks something like:
          #   /my-favorite-path/cifar10_train/model.ckpt-0,
          # extract global_step from it.
         # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return
      
      run_values = _RunValue()
      logger_hook = _LoggerHook()
      saver_hook = _saverHook(FLAGS.train_log_dir, 384, mysaver)
      logger_hook.begin()      
      saver_hook.begin()
      run_context = tf.train.SessionRunContext(None, restore_sess)
      for step in range(last_step):        
        logger_hook.before_run(run_context)
        saver_hook.before_run(run_context)
        run_values.results = restore_sess.run([loss, train_op])
        logger_hook.after_run(run_context, run_values)
        saver_hook.after_run(run_context, run_values)
        
#    with tf.train.MonitoredSession(
#        hooks=[tf.train.StopAtStepHook(last_step=last_step),
#               tf.train.NanTensorHook(loss),
#               _LoggerHook(),
#               _saverHook(FLAGS.train_log_dir, 384, mysaver)]) as mon_sess:
#      while not mon_sess.should_stop():      
#        mon_sess.run([train_op,accuracy_op])


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
  tf.gfile.MakeDirs(FLAGS.train_log_dir)
  last_step = int(math.ceil(FLAGS.num_examples_for_train / FLAGS.batch_size) * FLAGS.num_epochs)
  re_train(last_step)


if __name__ == '__main__':
  FLAGS = train_parser.parse_args()
  tf.app.run()
  
  