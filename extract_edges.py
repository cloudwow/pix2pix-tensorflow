import scipy.misc
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
def extract_image(filename):
    img = cv2.imread("simpsons/"+filename,cv2.IMREAD_COLOR)

    ###
    # lower mask (0-10)
    img_hsv = img
    lower_red = np.array([0,0,0])
    upper_red = np.array([255,30,255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)


    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask==0)] = [255,255,255]
#    cv2.imshow('image1',output_img)
   
   ### 
    edges = cv2.Canny(img,100,200)

    edges = cv2.bitwise_not(edges)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    print(filename)
    scipy.misc.imsave('simpsons_edges/'+filename, edges)    

    


# extract_image("s01e01_53.jpg")
for filename in os.listdir("simpsons"):
  extract_image(filename)  
