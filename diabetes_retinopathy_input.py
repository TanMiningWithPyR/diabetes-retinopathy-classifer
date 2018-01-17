"""Routine for decoding the diabetes_retinopathy photo format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import argparse

import pandas as pd
import tensorflow as tf

# Process images of this size. Note that this differs from the original diabetes retinopathy 
# image size, which are differece of each images bigger than 512 * 512. 
# If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 512

input_parser = argparse.ArgumentParser(add_help=False)
# Global constants describing the diabetes retinopathy data set.
input_parser.add_argument('--data_dir', type=str, default="D:\\kaggle\\detection\\train_tensorflow_512",
                    help='Path to the diabetes_retinopathy data directory.')
input_parser.add_argument('--labels_file', type=str, default="D:\\kaggle\\detection\\trainLabels\\trainLabels.csv",
                    help='Path to the diabetes_retinopathy labels file.')
input_parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of images to process in a batch.')
input_parser.add_argument('--num_examples_for_train', type=int, default=12288,
                    help='num examples per epoch for train.')
input_parser.add_argument('--num_examples_for_eval', type=int, default=5248,
                    help='num examples per epoch for evaluation.')
input_parser.add_argument('--num_epochs', type=int, default=60,
                    help='num epochs for train.')
input_parser.add_argument('--imbalance_ratio', type=int, default=1,
                    help='positive // negative')
input_parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

NUM_CLASSES = 5

def read_label(labels_file):
  df_labels = pd.read_csv(labels_file)
  patient = df_labels.image.str.split('_', expand=True)
  patient.columns=['patient', 'l_r']
  df_labels = pd.concat([df_labels,patient], axis=1)
  df_labels = df_labels.pivot('patient', 'l_r', 'level')
  return df_labels
  
def _transform_image(tf_filename, random_op=True):   
  tf_image_string = tf.read_file(tf_filename)
  tf_image_decoded = tf.image.decode_image(tf_image_string)  
  tf_distorted_image = tf_image_decoded
  # Image processing for training the network. Note the many random
  # distortions applied to the image.
  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # NOTE: since per_image_standardization zeros the mean and makes
  # the stddev unit, this likely has no effect (see tensorflow#1458).
  if random_op:
    tf_distorted_image = tf.image.random_contrast(tf_distorted_image, lower=0.2, upper=1.8)
#    # rotate
#    tf_distorted_image = _random_rotate(tf_distorted_image)
    tf_distorted_image = tf.image.random_brightness(tf_distorted_image, max_delta=0.2)
    tf_distorted_image = tf.image.random_saturation(tf_distorted_image, lower=0.9, upper=2)
    tf_distorted_image = tf.image.random_hue(tf_distorted_image, max_delta=0.02)
  # Subtract off the mean and divide by the variance of the pixels.
  # tf_distorted_image = tf.image.per_image_standardization(tf_distorted_image)
  tf_distorted_image = tf_distorted_image / 255
  return tf_distorted_image

def _parse_function_distorted(tf_patient_path,tf_label,tf_patient,tf_left_indication,tf_right_indication):  
  tf_patients_left_path = tf.string_join([tf_patient_path,tf.constant('left.jpeg')],separator='\\')
  tf_patients_right_path = tf.string_join([tf_patient_path, tf.constant('right.jpeg')],separator='\\')
  tf_left_image = _transform_image(tf_patients_left_path,random_op=True)
  tf_right_image = _transform_image(tf_patients_right_path,random_op=True)
  tf_left_label,tf_right_label = tf_label[0],tf_label[1]
  return tf_left_image, tf_left_label, tf_right_image, tf_right_label, tf_patient, tf_left_indication, tf_right_indication 

def _parse_function_normal(tf_patient_path,tf_label,tf_patient,tf_left_indication,tf_right_indication):  
  tf_patients_left_path = tf.string_join([tf_patient_path,tf.constant('left.jpeg')],separator='\\')
  tf_patients_right_path = tf.string_join([tf_patient_path, tf.constant('right.jpeg')],separator='\\')
  tf_left_image = _transform_image(tf_patients_left_path,random_op=False)
  tf_right_image = _transform_image(tf_patients_right_path,random_op=False)
  tf_left_label,tf_right_label = tf_label[0],tf_label[1]
  return tf_left_image, tf_left_label, tf_right_image, tf_right_label, tf_patient, tf_left_indication, tf_right_indication 
  
def distorted_inputs(data_dir, labels_file,  batch_size, imbalance_ratio=1, repeat=None):
  """Construct distorted input for diabetes retinopathy training using the Reader ops.

  Args:
    data_dir: Path to the diabetes retinopathy data directory.
    labels_file: Path to the label_file(.csv) for each eyes.
    batch_size: Number of images per batch.
    imbalance_ratio: Number of positive // Number of negative 
    repeat: how many times this dataset repeats during training. The default behavior is for the elements to be repeated indefinitely    

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  patients = os.listdir(data_dir)
  df_labels = read_label(labels_file)
  # oversample and shuffle
  bad_patients = [patient for patient in patients if sum(list(df_labels.ix[patient].values)) > 0 ]
  oversamples = patients + (imbalance_ratio - 1) * bad_patients
  num_step_in_epoch = len(oversamples) // batch_size
  num_examples_in_epoch = num_step_in_epoch * batch_size
  print("Training samples number after oversample: " + str(num_examples_in_epoch))
  random.shuffle(oversamples)
  patients_by_oversample = oversamples[0:num_examples_in_epoch]
  
  patient_paths = [os.path.join(data_dir, patient) for patient in patients_by_oversample]
  labels = [list(df_labels.ix[patient].values) for patient in patients_by_oversample]

  pair_eyes_count = len(patients_by_oversample)
  left_indications = pair_eyes_count * [(1.0, 0.0)]
  right_indications = pair_eyes_count * [(0.0, 1.0)]

  tf_patient_paths = tf.constant(patient_paths)
  tf_labels = tf.constant(labels)
  tf_patients = tf.constant(patients_by_oversample)
  tf_left_indications = tf.constant(left_indications)
  tf_right_indications = tf.constant(right_indications)
  
  tf_dataset = tf.data.Dataset.from_tensor_slices((tf_patient_paths, tf_labels, tf_patients, tf_left_indications, tf_right_indications))
  tf_dataset = tf_dataset.map(_parse_function_distorted)
  tf_dataset = tf_dataset.shuffle(buffer_size=32)
  tf_dataset = tf_dataset.batch(batch_size)
  tf_dataset = tf_dataset.repeat(repeat)
  tf_iterator = tf_dataset.make_one_shot_iterator()
  tf_next_left_image,tf_next_left_label, \
  tf_next_right_image,tf_next_right_label, \
  tf_next_patient, tf_next_left_indication, tf_next_right_indication = tf_iterator.get_next()  
  examples_num = tf_next_patient.shape[0].value
  return tf_next_left_image,tf_next_left_label, \
         tf_next_right_image,tf_next_right_label, \
         tf_next_patient, examples_num, \
         tf_next_left_indication,tf_next_right_indication

def inputs(data_dir, labels_file, batch_size):
  """Construct input for diabetes retinopathy evaluation using the Reader ops.

  Args:    
    data_dir: Path to the diabetes retinopathy data directory.
    labels_file: Path to the label_file(.csv) for each eyes
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  patients = os.listdir(data_dir)
  patient_paths = [os.path.join(data_dir,patient) for patient in patients]

  df_labels = read_label(labels_file)  
  labels = [list(df_labels.ix[patient].values) for patient in patients]

  pair_eyes_count = len(patients)
  left_indications = pair_eyes_count * [(1.0, 0.0)]
  right_indications = pair_eyes_count * [(0.0, 1.0)]

  tf_patient_paths = tf.constant(patient_paths)
  tf_labels = tf.constant(labels)
  tf_patients = tf.constant(patients)
  tf_left_indications = tf.constant(left_indications)
  tf_right_indications = tf.constant(right_indications)
  
  tf_dataset = tf.data.Dataset.from_tensor_slices((tf_patient_paths, tf_labels, tf_patients, tf_left_indications, tf_right_indications))
  tf_dataset = tf_dataset.map(_parse_function_normal)
  tf_dataset = tf_dataset.batch(batch_size)
  tf_iterator = tf_dataset.make_one_shot_iterator()
  tf_next_left_image,tf_next_left_label, \
  tf_next_right_image,tf_next_right_label, \
  tf_next_patient, tf_next_left_indication, tf_next_right_indication = tf_iterator.get_next()  
  return tf_next_left_image,tf_next_left_label,tf_next_right_image,tf_next_right_label,tf_next_patient,tf_next_left_indication,tf_next_right_indication
  
if __name__ == '__main__':
  labels_file = "D:\\kaggle\\detection\\trainLabels\\trainLabels.csv"
  data_dir = "D:\\kaggle\\detection\\trans_tensorflow\\train"
#  tf_next_left_image,tf_next_left_label, \
#  tf_next_right_image,tf_next_right_label, \
#  tf_next_patient, examples_num, tf_next_left_indication, tf_next_right_indication = distorted_inputs(data_dir,labels_file, 32, 1)  
  file_name = "D:\\kaggle\\detection\\train_tensorflow_512\\test\\17\\left.jpeg"
  import matplotlib.pyplot as plt
  sess = tf.Session()
#  left,right=sess.run((tf_next_left_image,tf_next_right_image))
#  plt.imshow(left[0])
#  pic = _transform_image(file_name, random_op=True)
  pic = tf.read_file(file_name)
  pic = tf.image.decode_jpeg(pic)
  pic = tf.reshape(pic, shape=[512,512,3])

  from scipy import misc 
  def _random_rotate(tf_image, max_angle=10):
    random_angle = random.randint(-max_angle, max_angle) * 3.14 / 180
    image_rotate = misc.imrotate(tf_image, random_angle, 'bicubic')
    return image_rotate
  p = sess.run(pic) 
  plt.imshow(p)
  
  
  
  
  
  
  