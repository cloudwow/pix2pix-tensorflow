import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from scipy import misc

def warp_it(image):
    import numpy as np

    from skimage.transform import PiecewiseAffineTransform, warp
    from scipy import misc
    from PIL import Image
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    
    # print(str(src))
    # print("00000000000000000000000000000000000000000000000000")
    # print(str(src[:, 1]))
    # print("00000000000000000000000000000000000000000000000000")

    # print(str(src[:, 0]))

    # # add sinusoidal oscillation to coordinates
    # dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 16
    # dst_cols = src[:, 0] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 16
    # dst_rows += 8
    # dst = np.vstack([dst_cols, dst_rows]).T
    from random import randint
    dst = src.copy()
    for i in range(dst.shape[0]):
        x=dst[i][0]
        y=dst[i][1]
        
        dst[i][0]+= randint(0,15)
        dst[i][1]+= randint(0,15)
         
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] -1.5 * 16
    out_cols = cols
    out = warp(image, tform, output_shape=(rows, cols))
    from skimage import img_as_ubyte

    return img_as_ubyte(out)
def add_gauss_noise(image):
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
def add_salt_and_pepper_noise(image):
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                                                  for i in image.shape]
        out[coords] = 0
        return out
image = misc.imread("../simpsons//s08e10_92.jpg")
image = add_gauss_noise(image)
fig, ax = plt.subplots()
from skimage import img_as_float

ax.imshow(img_as_float(image))
plt.show()
