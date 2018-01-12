"""A binary to train diabetes_retinopathy using a single GPU.

Accuracy:
diabetes_retinopathy_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 32.

System          | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
GeForce GTX 1070| 0.35-0.60              | 

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

train_parser.add_argument('--train_log_dir', type=str, default='D:\\AlanTan\\CNN\\diabetes_retinopathy_tensorflow\\diabetes_retinopathy_classifer_tensorflow_train',
                    help='Directory where to write event logs and checkpoint.')

train_parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

train_parser.add_argument('--log_frequency', type=int, default=-1,
                    help='How often to log results to the console.-1 is one epoch')

def train(last_step):
  """Train diabete_retinopathy for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    
    # Get images and labels for diabete_retinopathy.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    #with tf.device('/cpu:0'):
    left_images, left_labels, right_images, right_labels, patients, examples_num, left_indication, right_indication = \
    diabetes_retinopathy.distorted_inputs(FLAGS)
    
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
    
    # Calculate accuracy.
    accuracy_op = diabetes_retinopathy.eval_train_data(logits, labels)

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
        return tf.train.SessionRunArgs([loss,accuracy_op])  # Asks for loss and accuracy value.

      def after_run(self, run_context, run_values):          
        if self._step % self._log_frequency == self._log_frequency - 1 :
          current_time = time.time()
          duration = current_time - self._start_time

          loss_value = run_values.results[0]
          accuracy_value = run_values.results[1]
          steps_per_sec = self._log_frequency / duration
          sec_per_sec = float(duration / self._log_frequency)
          format_str = ('%s: step %d, epoch %d end, loss = %.3f, accuracy_value = %.3f (%.3f steps/sec; %.3f sec/step)')
          print (format_str % (datetime.now(), self._step, self._epoch, loss_value, accuracy_value,
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
        self._min_loss_arr = np.array(5 * [10.0 ** 5])
        
      def before_run(self, run_context):
        self._step += 1
        self._min_loss = np.max(self._min_loss_arr)
        self._max_ind = np.argmax(self._min_loss_arr)
        return tf.train.SessionRunArgs([loss, accuracy_op])
    
      def after_run(self, run_context, run_values):
        if run_values.results[0] < self._min_loss:
          if self._step % self._save_steps == self._save_steps - 1:          
            self._min_loss_arr[self._max_ind] = run_values.results[0]
            self._saver.save(run_context.session, self._save_path, self._step) 
            format_str = ('%s: step %d, loss = %.3f, accuracy_value = %.3f, weights was saved')
            print (format_str % (datetime.now(), self._step, run_values.results[0], run_values.results[1]))  
          elif self._step % 20 == 0:
            format_str = ('%s: step %d, loss = %.5f, accuracy_value = %.3f')
            print (format_str % (datetime.now(), self._step, run_values.results[0], run_values.results[1]))             
          
    mysaver=tf.train.Saver(max_to_keep=5)
    
    with tf.train.MonitoredTrainingSession(        
        checkpoint_dir=FLAGS.train_log_dir,
        save_checkpoint_secs=None,
        hooks=[tf.train.StopAtStepHook(last_step=last_step),
               tf.train.NanTensorHook(loss),
               _LoggerHook(),
               _saverHook(FLAGS.train_log_dir, 384, mysaver)],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run([train_op,accuracy_op])


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
  tf.gfile.MakeDirs(FLAGS.train_log_dir)
  last_step = int(math.ceil(FLAGS.num_examples_for_train / FLAGS.batch_size) * FLAGS.num_epochs)
  train(last_step)


if __name__ == '__main__':
  FLAGS = train_parser.parse_args()
  tf.app.run()
