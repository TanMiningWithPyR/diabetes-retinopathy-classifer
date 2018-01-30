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
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import diabetes_retinopathy

eval_parser = argparse.ArgumentParser(parents=[diabetes_retinopathy.model_parser], add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

eval_parser.add_argument('--eval_log_dir', type=str, default='D:\\AlanTan\\CNN\\diabetes_retinopathy_tensorflow\\diabetes_retinopathy_classifer_tensorflow_eval',
                    help='Directory where to write event logs.')

eval_parser.add_argument('--checkpoint_dir', type=str, default='D:\\AlanTan\\CNN\\diabetes_retinopathy_tensorflow\\diabetes_retinopathy_classifer_tensorflow_restore_train',
                    help='Directory where to read model checkpoints.')

eval_parser.add_argument('--eval_interval_secs', type=int, default=60*20,
                    help='How often to run the eval.')

eval_parser.add_argument('--run_once', type=bool, default=True,
                    help='Whether to run eval only once.')


def eval_once(saver, top_k_op, debug_op):
  """Run Eval once.

  Args:
    saver: Saver.
    top_k_op: Top K op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
#      # Assuming model_checkpoint_path looks something like:
#      #   /my-favorite-path/cifar10_train/model.ckpt-0,
#      # extract global_step from it.
#      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    list_predicts = []
    num_iter = int(math.ceil(FLAGS.num_examples_for_eval / FLAGS.batch_size))
    print(str(num_iter) + " iteration Totally. now: " + str(datetime.now()))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size
    step = 0
    while step < num_iter:
      if step % 10 == 0:
        print("The " + str(step) + " step...")          
      predictions, df_log = sess.run([top_k_op, debug_op]) # debug_op     
      
      patients = pd.DataFrame(df_log[0], columns=['patients'])
      labels = pd.DataFrame(df_log[1],columns=['labels'])
      logits = pd.DataFrame(df_log[2],columns=['L0','L1','L2','L3','L4'])
      predicts = pd.DataFrame(df_log[3],columns=['predicts'])
      df_predict = pd.concat([patients,labels,logits,predicts],axis=1)
      list_predicts.append(df_predict)
      true_count += np.sum(predictions)
      step += 1

   # Compute precision @ 1.
    precision = true_count / (total_sample_count * 2)
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        
    # output predict dataframe
    df_predicts = pd.concat(list_predicts)
    df_predicts.to_csv(os.path.join(FLAGS.eval_log_dir,"predicts.csv"))

def evaluate():
  """Eval diabetes_retinopathy for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for diabetes_retinopathy.
    with tf.device('/cpu:0'):
      left_images, left_labels, right_images, right_labels, patients,left_indication, right_indication = \
      diabetes_retinopathy.inputs(FLAGS)

    labels = tf.concat([left_labels, right_labels], 0)
    images = tf.concat([left_images, right_images], 0)
    indication = tf.concat([left_indication, right_indication], 0)  
    patients = tf.concat([patients,patients],0)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = diabetes_retinopathy.inference(images, indication, FLAGS) 
    
    # predict label    
    predicts = tf.argmax(logits, axis=1)
    
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # predict value compare source label
    p_s_df_op =  [patients, labels, tf.nn.softmax(logits), predicts]
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        diabetes_retinopathy.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
#    saver = tf.train.Saver()

    while True:
      eval_once(saver, top_k_op, p_s_df_op)
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
