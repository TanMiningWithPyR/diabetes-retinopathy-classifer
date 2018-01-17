# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:54:55 2017

@author: admin
"""
# import standard module
import os
import argparse
import time
from multiprocessing import Pool as P_Pool
from multiprocessing.dummy import Pool as T_Pool
# import extand module
import scipy
#import cv2
from skimage import transform 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='C:\\Users\\tanalan\\Documents\\CNN image\\detection\\sample',
                    help='Path to the diabetes_retinopathy data directory.')
parser.add_argument('--target_dir', type=str, default='C:\\Users\\tanalan\\Documents\\CNN image\\detection\\resize',
                    help='Path to data after resized')
parser.add_argument('--image_size', type=int, default=512,
                    help='512*512 pixel')
parser.add_argument('--Pro_or_Thd', type=str, default=None,
                    help='multiprocess or multithread')

def read_image(filename):
  #return a numpy array
  image = scipy.misc.imread(filename)
  return image

def save_image(image, filename):
  scipy.misc.imsave(filename,image)  

def crop(image):  
  x_m,y_m = image.shape[0]//2 + 1,image.shape[1]//2 + 1
  x_MiddleLine_RGBsum = image[x_m,:,:].sum(1)
  r_y = (x_MiddleLine_RGBsum>x_MiddleLine_RGBsum.mean()/500).sum()//2 + 1
  y_MiddleLine_RGBsum = image[:,y_m,:].sum(1)
  r_x = (y_MiddleLine_RGBsum>y_MiddleLine_RGBsum.mean()/500).sum()//2 + 1
  image_crop = image[x_m-r_x:x_m+r_x,y_m-r_y:y_m+r_y,:]
  return image_crop

def resize(image,image_size):
  image_resize = transform.resize(image,
        (image_size,image_size),
        mode='reflect')
  return image_resize

def grey_map(image):
  image_grey_map = cv2.addWeighted(image_resize,4,
                                     cv2.GaussianBlur(image_resize,(0,0),10),
                                     -4,128)
  return image_grey_map

def do_one(filename, flags):
  src_filename = os.path.join(flags.data_dir, filename)
  image = read_image(src_filename)
  image = crop(image)
  image = resize(image, flags.image_size)
#  image = grey_map(image)
  tar_filename = os.path.join(flags.target_dir, filename)
  save_image(image, tar_filename)  
  return(image)

def do_all(flags):  
  filenames = os.listdir(flags.data_dir)
  for filename in filenames:
    do_one(filename, flags)
               
def do_all_multiprocess(flags):
  filenames = os.listdir(flags.data_dir)
  p = P_Pool(4) 
  for filename in filenames:
    p.apply_async(do_one, args=(filename,flags,))
  print('Waiting for all subprocesses done...')
  p.close()
  p.join()
  print('All subprocesses done.') 

def do_all_multithread(flags):
  filenames = os.listdir(flags.data_dir)
  p = T_Pool(4) 
  for filename in filenames:
    p.apply_async(do_one, args=(filename,flags,))
  print('Waiting for all subthread done...')
  p.close()
  p.join()
  print('All subthread done.')   

if __name__ == '__main__':
  FLAGS = parser.parse_args()
  if FLAGS.Pro_or_Thd == None:
    start = time.time()
    do_all(FLAGS)
    end = time.time()
    print('Task runs %0.5f seconds.' % ((end - start)))
  elif FLAGS.Pro_or_Thd == 'Pro':
    start = time.time()
    do_all_multiprocess(FLAGS)       
    end = time.time()
    print('Task runs %0.5f seconds.' % ((end - start)))
  elif FLAGS.Pro_or_Thd == 'Thd':
    start = time.time()
    do_all_multithread(FLAGS)       
    end = time.time()
    print('Task runs %0.5f seconds.' % ((end - start)))  
                
    