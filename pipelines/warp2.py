import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from scipy import misc

img = misc.imread("./simpsons/s11e09_201.jpg")
print("shape[0]: " + str(img.shape[0]))
print("shape[1]: " + str(img.shape[1]))
print("shape[2]: " + str(img.shape[2]))
A = img.shape[0] / 3.0
w = 2.0 / img.shape[1]

shift = lambda x: A * np.sin(1.1*np.pi*x * w)
for channel in range(3):
    for i in range(img.shape[0]):
        img[:,i,channel] = np.roll(img[:,i,channel], int(shift(i)))

plt.imshow(img)
plt.show()
        
