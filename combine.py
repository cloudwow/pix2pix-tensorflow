import scipy.misc
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)

    return img[starty:starty+cropy,startx:startx+cropx]
def combine(filename):
    img_left = cv2.imread("simpsons_edges/"+filename,cv2.IMREAD_COLOR)
    img_left = img_left[0:383, 64:447,0:3]
    img_left = cv2.resize(img_left, (256, 256)) 
    img_right = cv2.imread("simpsons/"+filename,cv2.IMREAD_COLOR)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

    img_right = img_right[0:383, 64:447,0:3]
    img_right=cv2.resize(img_right, (256, 256)) 

    new_img = np.zeros((256,512,3), np.uint8)
    x_offset=256
    y_offset=0
    new_img[y_offset:img_right.shape[0], x_offset:256+img_right.shape[1]] = img_right
    x_offset=0
    y_offset=0
    new_img[y_offset:img_left.shape[0], x_offset:img_left.shape[1]] = img_left
    print(filename)
#    cv2.imshow('image1',new_img)

#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    scipy.misc.imsave('simpsons_examples/'+filename.replace("jpg","png"), new_img)    


#combine("s01e01_53.jpg")
for filename in os.listdir("simpsons"):
  combine(filename)  
