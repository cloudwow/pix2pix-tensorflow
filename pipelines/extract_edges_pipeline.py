import ml_args as ml_args
import argparse
#import csv
#import datetime
import errno
#import io
import logging
import os
#import subprocess
#import sys
#import binascii
#import base64
#import six


from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from scipy import misc

import apache_beam as beam

from apache_beam.metrics import Metrics


try:
    try:
        from apache_beam.options.pipeline_options import PipelineOptions
    except ImportError:
        from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
        from apache_beam.utils.options import PipelineOptions


class ListFiles(beam.DoFn):

    def process(self, uri):
        logging.info("listing files in  "+str(uri));
        from tensorflow.python.lib.io import file_io

        
        for target_file in file_io.list_directory(uri):
            # logging.info("yielding "+target_file);
            yield uri+"/"+target_file;
        
      

class ReadImage(beam.DoFn):
    def __init__(self, noise):
        self.noise = noise
        self.error_count = Metrics.counter('main', 'errorCount')
        self.success_count = Metrics.counter('main', 'successCount')
        
    def save_np_image(self, np_image, destination):
        import PIL.Image
        final_image =  PIL.Image.fromarray(np_image)
        import io
        final_image_bytes = io.BytesIO()
        final_image.save(final_image_bytes, format='JPEG')
        result_bytes= final_image_bytes.getvalue()
        from apache_beam.io.gcp import gcsfilesystem
        
        file_system = gcsfilesystem.GCSFileSystem()
        file  = file_system.create(destination, 'image/jpeg')

        file.write(result_bytes)
        file.close()
    def add_gauss_noise(self, image):
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        import numpy as np

        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    def add_salt_and_pepper_noise(self, image):
        import numpy as np

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

    def warp_it(self, image):
        import numpy as np

        from skimage.transform import PiecewiseAffineTransform, warp
        from scipy import misc
        from PIL import Image
        rows, cols = image.shape[0], image.shape[1]

        src_cols = np.linspace(0, cols, 20) + np.random.uniform(-3,3,20)
        src_rows = np.linspace(0, rows, 10) + np.random.uniform(-3,3,10)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]

        from random import randint
        dst = src.copy()
        for i in range(dst.shape[0]):
            x=dst[i][0]
            y=dst[i][1]
        
            dst[i][0]+= randint(-8,8)
            dst[i][1]+= randint(-8,8)


        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)

        out_rows = image.shape[0] -1.5 * 16
        out_cols = cols
        out = warp(image, tform, output_shape=(rows, cols), mode='constant', cval=1.0)

        from skimage import img_as_ubyte

        return img_as_ubyte(out)

    def process(self, uri):
        import time
        the_time = long(time.time())
        from tensorflow.python.lib.io import file_io
        from tensorflow.python.framework import errors
        
        def _open_file_read_binary(uri):
            
            try:
                return file_io.FileIO(uri, mode='rb')
            except errors.InvalidArgumentError:
                return file_io.FileIO(uri, mode='r')
            
        try:
            with _open_file_read_binary(uri) as f:
                image_bytes = f.read()
                
        # A variety of different calling libraries throw different exceptions here.
        # They all correspond to an unreadable file so we treat them equivalently.
        except Exception as e:  # pylint: disable=broad-except
            logging.exception('Error processing image %s: %s', uri, str(e))
            self.error_count.inc()
            return

        self.success_count.inc();
        # Convert to desired format and output.
        import cv2
        import numpy as np
        nparr = np.fromstring(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        # fix weird color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# crop the center
        img = img[0:383, 64:447,0:3]
        # resize to 256 square
        img = cv2.resize(img, (512, 512)) 
        #        self.save_np_image(img, uri.replace("source","target"))
        original_image =img.copy()


#        img = cv2.imdecode(img, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

        
        #        print(str(img))
        #self.save_np_image(img, uri.replace("source","derp"))
        # img = cv2.imdecode(img, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

        
#        print(str(img))
        
 
        
        ###
        # lower mask (0-10)

        
      
        ###
        if(self.noise == "salt_and_pepper"):
            img=self.add_salt_and_pepper_noise(img)
        elif self.noise == "gauss":
            img=self.add_gauss_noise(img)

        edges = cv2.Canny(img,100,200)
        edges[0]=0
        edges[edges.shape[0]-1]=0
        
        edges = cv2.bitwise_not(edges)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        img2 = np.zeros_like(img)
        img2[:,:,0] = edges
        img2[:,:,1] = edges
        img2[:,:,2] = edges
        edges =img2
        edges = self.warp_it(edges)
        train_img = np.full((512,1024,3), 255, dtype=np.uint8)
        x_offset=512
        y_offset=0
        train_img[ y_offset:original_image.shape[0], x_offset:512+original_image.shape[1]] = original_image
        x_offset=0
        y_offset=0
        train_img[y_offset:edges.shape[0], x_offset:edges.shape[1]] = edges
        #        if random.randrange(10) >=8:
        self.save_np_image(train_img, uri.replace("source","train")
            .replace(".jpg","_distorted_"+str(the_time)+".jpg"))
    
def run(args):
#    pipeline_options = PipelineOptions.from_dictionary(vars(args))
    pipeline_options = PipelineOptions()
  
    with beam.Pipeline(options=pipeline_options) as p:
         _ = (p | beam.Create(["gs://" + args.gcs_bucket+"/"+args.topic+"/source_images"])
                | "list files" >> beam.ParDo(ListFiles())
                | "read image" >> beam.ParDo(ReadImage(args.noise)))

  
def main():
    args=ml_args.process_args()
    run(args)

if __name__ == '__main__':
    main()
