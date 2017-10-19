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

BASE_DIR="gs://pix2pixdata"
EPS = 1e-12

CROP_SIZE = 256

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train, global_step")

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2



def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image




def create_generator(num_generator_filters, generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, num_generator_filters]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, num_generator_filters, stride=2)
        layers.append(output)

    layer_specs = [
        num_generator_filters * 2, # encoder_2: [batch, 128, 128, num_generator_filters] => [batch, 64, 64, num_generator_filters * 2]
        num_generator_filters * 4, # encoder_3: [batch, 64, 64, num_generator_filters * 2] => [batch, 32, 32, num_generator_filters * 4]
        num_generator_filters * 8, # encoder_4: [batch, 32, 32, num_generator_filters * 4] => [batch, 16, 16, num_generator_filters * 8]
        num_generator_filters * 8, # encoder_5: [batch, 16, 16, num_generator_filters * 8] => [batch, 8, 8, num_generator_filters * 8]
        num_generator_filters * 8, # encoder_6: [batch, 8, 8, num_generator_filters * 8] => [batch, 4, 4, num_generator_filters * 8]
        num_generator_filters * 8, # encoder_7: [batch, 4, 4, num_generator_filters * 8] => [batch, 2, 2, num_generator_filters * 8]
        num_generator_filters * 8, # encoder_8: [batch, 2, 2, num_generator_filters * 8] => [batch, 1, 1, num_generator_filters * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (num_generator_filters * 8, 0.5),   # decoder_8: [batch, 1, 1, num_generator_filters * 8] => [batch, 2, 2, num_generator_filters * 8 * 2]
        (num_generator_filters * 8, 0.5),   # decoder_7: [batch, 2, 2, num_generator_filters * 8 * 2] => [batch, 4, 4, num_generator_filters * 8 * 2]
        (num_generator_filters * 8, 0.5),   # decoder_6: [batch, 4, 4, num_generator_filters * 8 * 2] => [batch, 8, 8, num_generator_filters * 8 * 2]
        (num_generator_filters * 8, 0.0),   # decoder_5: [batch, 8, 8, num_generator_filters * 8 * 2] => [batch, 16, 16, num_generator_filters * 8 * 2]
        (num_generator_filters * 4, 0.0),   # decoder_4: [batch, 16, 16, num_generator_filters * 8 * 2] => [batch, 32, 32, num_generator_filters * 4 * 2]
        (num_generator_filters * 2, 0.0),   # decoder_3: [batch, 32, 32, num_generator_filters * 4 * 2] => [batch, 64, 64, num_generator_filters * 2 * 2]
        (num_generator_filters, 0.0),       # decoder_2: [batch, 64, 64, num_generator_filters * 2 * 2] => [batch, 128, 128, num_generator_filters * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, num_generator_filters * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]




def run(target, is_chief, job_name, a):
    output_dir = "./output"

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
        batch_output = deprocess(create_generator(a.num_generator_filters,  preprocess(batch_input), 3))

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



