"""
Several functions:
1.collecting two eyes of one patient, put into one folder named as patients.
2.Dividing source data into two or three classes, for two classes, as train or test; for three classes , as train, train_eval, test.
3.rmove empty folders
"""

import os
import shutil
import argparse

import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Basic model parameters.

parser.add_argument('--data_dir', type=str, default='C:\\Users\\tanalan\\Documents\\CNN image\\detection\\sample',
                    help='Path to the diabetes_retinopathy data directory.')
parser.add_argument('--picture_type', type=str, default='jpeg',
                    help='the picture type, such as png, jpeg.')
parser.add_argument('--split_num', type=int, default=None, choices=[2, 3],
					       help='how many sub sets splitted from source data. ')
parser.add_argument('--reverse_op', type=bool, default=False,
                    help='reverse operation for the files moving.')
parser.add_argument('--rmdir', type=str, default=None,
                    help='Judge the folders below one folder whether is empty.')

def _rmdir(flags):
  data_dir = flags.rmdir
  patients = os.listdir(data_dir)
  patients_path = [os.path.join(data_dir,patient) for patient in patients]

  for patient_path in patients_path:
    if os.path.isdir(patient_path):
      if not os.listdir(patient_path):
         print(patient_path + " is empty! It will be remove")
         os.rmdir(patient_path)  

def _mkdir(path):
  try:
    os.mkdir(path)
  except FileExistsError:
    print("folder--(" + path +  ") exists")

def _collect_move(flags):
  filenames = os.listdir(flags.data_dir)
  filenames = [filename for filename in filenames if flags.picture_type in filename.split('.')]
  patients = set([filename.split('_')[0] for filename in filenames])
  for patient in patients:
    dst = os.path.join(flags.data_dir,patient)
    _mkdir(dst)
    src_left = os.path.join(flags.data_dir,patient + "_left.jpeg")
    src_right = os.path.join(flags.data_dir,patient + "_right.jpeg")		
    try:
      dst_left = os.path.join(dst,"left.jpeg")
      shutil.move(src_left,dst_left)
      dst_right = os.path.join(dst,"right.jpeg")
      shutil.move(src_right,dst_right)			
    except FileNotFoundError:
      print("Pictures of " + patient + "th patient are lost! It will be deleted")	
      try:					 
        os.remove(src_right)
      except FileNotFoundError: 
        os.remove(dst_left)
                
      try:	
        os.rmdir(dst)
      except:
        pass

def _reverse_op(flags):
  patients = [path for path in os.listdir(flags.data_dir) if os.path.isdir(os.path.join(flags.data_dir,path))]
  for patient in patients:
    src = os.path.join(flags.data_dir,patient)
    for eye in os.listdir(src):
      one_file = os.path.join(src,eye) 
      dst_one_file = os.path.join(flags.data_dir,patient + '_' +eye)
      shutil.move(one_file,dst_one_file)
      os.rmdir(src)	

def _split_train_eval(flags):
  all_patients = os.listdir(flags.data_dir)	
  patients_num = len(all_patients)
  arr_random = np.random.rand(patients_num)
  patients_dist = dict(zip(all_patients,arr_random))
  train_dir = os.path.join(flags.data_dir, 'train')	
  eval_dir = os.path.join(flags.data_dir, 'train_eval')
  test_dir = os.path.join(flags.data_dir, 'test')
  _mkdir(train_dir)
  _mkdir(test_dir)
  if flags.split_num==3:
    _mkdir(eval_dir)
    for patient in patients_dist:
      src = os.path.join(flags.data_dir, patient)
      if patients_dist[patient] < 0.7:
        dst = train_dir
        shutil.move(src, dst)
      elif patients_dist[patient] < 0.9:
        dst = eval_dir
        shutil.move(src, dst)
      else:
        dst = test_dir
        shutil.move(src, dst)
  elif flags.split_num==2:
    for patient in patients_dist:
      src = os.path.join(flags.data_dir, patient)
      if patients_dist[patient] < 0.7:
        dst = train_dir
        shutil.move(src, dst)
      else:
        dst = test_dir
        shutil.move(src, dst)
  else:
    print("split_num must be 2 or 3")		

def main(argv=None):
  if FLAGS.split_num!=None:
    _split_train_eval(FLAGS)
  if FLAGS.rmdir!=None:
    _rmdir(FLAGS)
  else:    
    if FLAGS.reverse_op:
      _reverse_op(FLAGS)
    else:
      _collect_move(FLAGS)


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  main()
