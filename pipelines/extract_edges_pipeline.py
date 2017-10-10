import ml_args as ml_args
import argparse
import csv
import datetime
import errno
import io
import logging
import os
import subprocess
import sys
import binascii
import base64
import six

import apache_beam as beam

from apache_beam.metrics import Metrics
from apache_beam.io.gcp import gcsfilesystem


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
  def __init__(self):
      self.error_count = Metrics.counter('main', 'errorCount')
      self.success_count = Metrics.counter('main', 'successCount')

  def process(self, uri):
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
      import PIL.Image
      final_image =  PIL.Image.fromarray(edges)

      final_image_bytes = io.BytesIO()
      final_image.save(final_image_bytes, format='PNG')
      result_bytes= final_image_bytes.getvalue()
      file_system = gcsfilesystem.GCSFileSystem()
      file  = file_system.create(uri.replace("source","edge"), 'image/png')

      file.write(result_bytes)
      file.close()

    
def run(args):
#    pipeline_options = PipelineOptions.from_dictionary(vars(args))
    pipeline_options = PipelineOptions()
  
    with beam.Pipeline(options=pipeline_options) as p:
         _ = (p | beam.Create(["gs://" + args.gcs_bucket+"/simpsons/source_images"])
                | "list files" >> beam.ParDo(ListFiles())
                | "read image" >> beam.ParDo(ReadImage()))

  
def main():
    args=ml_args.process_args()
    run(args)

if __name__ == '__main__':
    main()
