import scipy.misc
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint

for filename in os.listdir("simpsons_examples"):
   if filename == 'train':
       continue
   if filename == 'val':
       continue
   if filename == 'test':
       continue
   d=randint(0, 9)
   if d<3:
       os.rename("simpsons_examples/"+filename, "simpsons_examples/val/"+filename)
   elif d<4 :
       os.rename("simpsons_examples/"+filename, "simpsons_examples/test/"+filename)
   else:
       os.rename("simpsons_examples/"+filename, "simpsons_examples/train/"+filename)
