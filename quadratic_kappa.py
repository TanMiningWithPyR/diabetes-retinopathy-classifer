# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:21:05 2017

@author: tanalan
"""
import numpy as np
import tensorflow as tf

# numpy's version
def quadratic_kappa(y, t, eps=1e-15):
      # Assuming y and t are one-hot encoded!
      
      # An N-by-N matrix of weights, w, is calculated based on the difference between raters' scores:
      num_scored_items = y.shape[0]
      num_ratings = y.shape[1] 
      ratings_mat = np.tile(np.arange(0, num_ratings)[:, None], 
                            reps=(1, num_ratings))
      ratings_squared = (ratings_mat - ratings_mat.T) ** 2
      weights = ratings_squared / (float(num_ratings) - 1) ** 2
	
      # We norm for consistency with other variations.
      y_norm = y / (eps + y.sum(axis=1)[:, None])
	
      # The histograms of the raters.
      hist_rater_a = y_norm.sum(axis=0)
      hist_rater_b = t.sum(axis=0)
	
      # The confusion matrix.
      conf_mat = np.dot(y_norm.T, t)
	
      # The nominator.
      nom = np.sum(weights * conf_mat)
      expected_probs = np.dot(hist_rater_a[:, None], 
                              hist_rater_b[None, :])
      # The denominator.
      denom = np.sum(weights * expected_probs / num_scored_items)
	
      return 1 - nom / denom

# tensorflow's version      
def quadratic_kappa_on_one_hot_value(predict, real, examples_num, eps=1e-15):
  # Assuming y and real are one-hot encoded! 
  real = tf.cast(real, tf.float64)
  # An N-by-N matrix of weights, w, is calculated based on the difference between raters' scores:
  num_scored_items = examples_num
  num_ratings = predict.shape[1].value
  ratings_mat = tf.tile(tf.reshape(tf.constant(np.arange(0,num_ratings)),(num_ratings,-1)),multiples=(1, num_ratings))
  ratings_squared = (ratings_mat - tf.transpose(ratings_mat)) ** 2
  weights = ratings_squared / (num_ratings - 1) ** 2
	
  # We norm for consistency with other variations.
  predict_norm = tf.cast(predict / (eps + tf.reduce_sum(predict,axis=1)[:,None]), tf.float64)	
  # The histograms of the raters.
  hist_rater_a = tf.reduce_sum(predict_norm, axis=0)
  hist_rater_b = tf.reduce_sum(real, axis=0)
	
  # The confusion matrix.
  conf_mat = tf.tensordot(tf.transpose(predict_norm), tf.cast(real, tf.float64), axes=1)
	
  # The nominator.
  nom = tf.reduce_sum(weights * conf_mat)
  expected_probs = tf.tensordot(hist_rater_a[:, None], 
                          hist_rater_b[None, :], axes=1)
  # The denominator.
  denom = tf.reduce_sum(weights * expected_probs / num_scored_items)
	
  return tf.cast(1 - nom / denom, tf.float32)
      
if __name__ == '__main__':
  y = np.array([[0.4,0.3,0.2,0.1,0.0],[0.0,0.1,0.2,0.3,0.4]])
  t = np.array([[1,0,0,0,0],[0,0,1,0,0]])
  qk = quadratic_kappa(y,t)
  predict = tf.constant(y)
  real = tf.constant(t)
  qk_tf = quadratic_kappa_on_one_hot_value(predict,real,2)
  with tf.Session() as sess:
    print([sess.run(qk_tf),qk_tf])