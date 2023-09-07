import os 
import glob
import random

images_dir='data/images'
annotpath_dir='data/annots'
random.seed(42)

imagepath_list = glob.glob(os.path.join(images_dir, '*.jpg'))
random.shuffle(imagepath_list)

padding = len(str(len(imagepath_list)))

for n, filepath in enumerate(imagepath_list, 1):
    os.rename(filepath, os.path.join(images_dir, '{:>0{}}.jpg'.format(n, padding)))
        
annotpath_list = glob.glob(os.path.join(annotpath_dir, '*.xml'))
random.shuffle(annotpath_list)

for m, filepath in enumerate(annotpath_list, 1):
    os.rename(filepath, os.path.join(annotpath_dir, '{:>0{}}.xml'.format(m, padding)))
    
    