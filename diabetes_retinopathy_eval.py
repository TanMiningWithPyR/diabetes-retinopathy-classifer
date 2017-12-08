"""Evaluation for diabetes_retinopathy.

Accuracy:
diabetes_retinopathy_train.py achieves *num* accuracy after *num* steps (*num*  epochs
of data) as judged by diabetes_retinopathy_eval.py.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import argparse
import time

import numpy as np
import tensorflow as tf

import diabetes_retinopathy

eval_parser = argparse.ArgumentParser(parents=[diabetes_retinopathy.model_parser], add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

eval_parser.add_argument('--eval_log_dir', type=str, default='D:\\AlanTan\\CNN\\diabetes_retinopathy_tensorflow\\diabetes_retinopathy_classifer_tensorflow_eval',
                    help='Directory where to write event logs.')

eval_parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

eval_parser.add_argument('--checkpoint_dir', type=str, default='D:\\AlanTan\\CNN\\diabetes_retinopathy_tensorflow\\diabetes_retinopathy_classifer_tensorflow_train',
                    help='Directory where to read model checkpoints.')

eval_parser.add_argument('--eval_interval_secs', type=int, default=60*20,
                    help='How often to run the eval.')

eval_parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples_for_eval / FLAGS.batch_size))
      print(str(num_iter) + " iteration Totally. now: " + str(datetime.now()))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        if step % 10 == 0:
          print("The " + str(step) + " step...")
        try:
          predictions = sess.run([top_k_op])
          true_count += np.sum(predictions)
          step += 1
        except tf.errors.OutOfRangeError:
          break

      # Compute precision @ 1.
      precision = true_count / (total_sample_count * 2)
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval diabetes_retinopathy for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for diabetes_retinopathy.
    with tf.device('/cpu:0'):
      left_images, left_labels, right_images, right_labels, patients = \
      diabetes_retinopathy.inputs(FLAGS.eval_data, FLAGS.data_dir, FLAGS.labels_file, FLAGS.batch_size, FLAGS.use_fp16)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    left_logits, right_logits = diabetes_retinopathy.inference(left_images, right_images, 
                                                               FLAGS.batch_size, FLAGS.use_fp16)

    # Calculate predictions.
    left_top_k_op = tf.nn.in_top_k(left_logits, left_labels, 1)
    right_top_k_op = tf.nn.in_top_k(right_logits, right_labels, 1)
    top_k_op = tf.concat([left_top_k_op, right_top_k_op], axis=-1)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        diabetes_retinopathy.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_log_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument  
  if tf.gfile.Exists(FLAGS.eval_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_log_dir)
  tf.gfile.MakeDirs(FLAGS.eval_log_dir)
  evaluate()


if __name__ == '__main__':
  FLAGS = eval_parser.parse_args()
  tf.app.run()
