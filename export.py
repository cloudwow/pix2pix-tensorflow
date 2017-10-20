from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.lib.io import file_io
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import threading
import myp2p.model as model
BASE_DIR="gs://pix2pixdata"
EPS = 1e-12

CROP_SIZE = 256

def run(target, is_chief, job_name, a):
    output_dir = "./export"

    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)


    for k, v in a._get_kwargs():
        print(k, "=", v)

    if a.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint
    # disable these features in test mode
    a.scale_size = CROP_SIZE
    a.flip = False

    input = tf.placeholder(tf.string, shape=[1])
    input_data = tf.decode_base64(input[0])
    input_image = tf.image.decode_png(input_data)

    # remove alpha channel if present
    input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
    # convert grayscale to RGB
    input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

    input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
    input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
    batch_input = tf.expand_dims(input_image, axis=0)

    with tf.variable_scope("generator"):
        batch_output = model.deprocess(
            model.create_generator(a.num_generator_filters,
                                   model.preprocess(batch_input), 3))

    output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
    if a.output_filetype == "jpeg":
        output_data = tf.image.encode_jpeg(output_image, quality=80)
    else:
        output_data = tf.image.encode_png(output_image)

    output = tf.convert_to_tensor([tf.encode_base64(output_data)])

    key = tf.placeholder(tf.string, shape=[1])
    inputs = {
        "key": key.name,
        "input": input.name
    }
    tf.add_to_collection("inputs", json.dumps(inputs))
    outputs = {
        "key":  tf.identity(key).name,
        "output": output.name,
    }
    tf.add_to_collection("outputs", json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

            

    with tf.Session() as sess:
        print("monitored session created.")
        sess.run(init_op)
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        restore_saver.restore(sess, checkpoint)
        # ready to process image
        print("exporting model")
        export_saver.export_meta_graph(filename=os.path.join(output_dir, "export.meta"))
        export_saver.save(sess, os.path.join(output_dir, "export"), write_meta_graph=False)
        



                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", help="e.g. simpson. should match folder name in GCS")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--output_dir", default=None, help="directory with checkpoint to resume training from or use for testing")


    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--num_generator_filters", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--num_discriminator_filters", type=int, default=64, help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
    parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
    parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
    parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')

    a, unknown = parser.parse_known_args()

    tf.logging.set_verbosity(a.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[a.verbosity] / 10)

    # export options

    del a.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        run('', True, "local", a=a)
    else :
        tf_config_json = json.loads(tf_config)

        cluster = tf_config_json.get('cluster')
        job_name = tf_config_json.get('task', {}).get('type')
        task_index = tf_config_json.get('task', {}).get('index')

        # If cluster information is empty run local
        if job_name is None or task_index is None:
          run('', False, job_name ,a=a)
        else:
            cluster_spec = tf.train.ClusterSpec(cluster)
            server = tf.train.Server(cluster_spec,
                                       job_name=job_name,
                                       task_index=task_index)

            # Wait for incoming connections forever
            # Worker ships the graph to the ps server
            # The ps server manages the parameters of the model.
            #
            # See a detailed video on distributed TensorFlow
            # https://www.youtube.com/watch?v=la_M6bCV91M
            if job_name == 'ps':
                server.join()           
            elif job_name in ['master', 'worker']:
                run(server.target, job_name == 'master', job_name, a=a)



