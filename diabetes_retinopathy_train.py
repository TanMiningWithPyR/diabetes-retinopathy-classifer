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
import time
import math
import argparse
import tensorflow as tf

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
    with tf.device('/cpu:0'):
      left_images, left_labels, right_images, right_labels, patients, left_indication, right_indication = \
      diabetes_retinopathy.distorted_inputs(FLAGS)

    # Build a Graph that computes the logits predictions from the
    # inference model.    
    left_logits, right_logits = diabetes_retinopathy.inference(left_images, right_images, 
                                                               left_indication, right_indication, FLAGS)

    # Calculate loss.
    loss = diabetes_retinopathy.loss(left_logits, right_logits, left_labels, right_labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = diabetes_retinopathy.train(loss, global_step, FLAGS)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if FLAGS.log_frequency == -1:   # how many steps in one epoch 
          log_frequency = int(math.ceil(FLAGS.num_examples_for_train / FLAGS.batch_size))
        else:
          log_frequency = FLAGS.log_frequency
          
        if self._step % log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(        
        checkpoint_dir=FLAGS.train_log_dir,
        hooks=[tf.train.StopAtStepHook(last_step=last_step),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
  tf.gfile.MakeDirs(FLAGS.train_log_dir)
  last_step = int(math.ceil(FLAGS.num_examples_for_train / FLAGS.batch_size) * FLAGS.num_epochs)
  train(last_step)


if __name__ == '__main__':
  FLAGS = train_parser.parse_args()
  tf.app.run()
