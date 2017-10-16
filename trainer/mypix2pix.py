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

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
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


def load_examples(input_dir, mode, scale_size, batch_size):
    input_dir = input_dir+"/"+mode+"_images"
    print("input dir for mode "+mode+" is "+input_dir)
    input_paths = []
    input_paths.extend(file_io.get_matching_files(input_dir+"/*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//2,:])
        b_images = preprocess(raw_input[:,width//2:,:])

        inputs, targets = [a_images, b_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


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


def create_model(inputs, targets,
                 num_generator_filters,
                 num_discriminator_filters,
                 gan_weight,
                 l1_weight,
                 lr,
                 beta1):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, num_discriminator_filters]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, num_discriminator_filters, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, num_discriminator_filters] => [batch, 64, 64, num_discriminator_filters * 2]
        # layer_3: [batch, 64, 64, num_discriminator_filters * 2] => [batch, 32, 32, num_discriminator_filters * 4]
        # layer_4: [batch, 32, 32, num_discriminator_filters * 4] => [batch, 31, 31, num_discriminator_filters * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = num_discriminator_filters * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, num_discriminator_filters * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(num_generator_filters,  inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
        global_step=global_step
    )


def save_images(fetches, output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            file_io.FileIO(output_path, "wb").write(contents);
        filesets.append(fileset)
    return filesets


def append_index(image_directory, filesets, step=False):
    index_loc = image_directory+"/index.html"
    if file_io.FileIO.exists(index_loc):
        index_html = "<html><body><table><tr>\r\n"
    else:
        index_html =  file_io.FileIO(index_loc, "r").read();
        
    if step:
        index_html+="<th>step</th>"
        index_html+="<th>name</th><th>input</th><th>output</th><th>target</th></tr>"

    for fileset in filesets:
        index_html+="<tr>"

        if step:
            index_html += "<td>%d</td>" % fileset["step"]
        index_html += "<td>%s</td>" % fileset["name"]

        for kind in ["inputs", "outputs", "targets"]:
            index_html += "<td><img src='images/%s'></td>" % fileset[kind]

        index_html += "</tr>"
    file_io.FileIO(index_loc, "w").write(index_html);


class EvaluationRunHook(tf.train.SessionRunHook):
  """EvaluationRunHook performs continuous evaluation of the model.

  Args:
    checkpoint_dir (string): Dir to store model checkpoints
    metric_dir (string): Dir to store metrics like accuracy and auroc
    graph (tf.Graph): Evaluation graph
    eval_frequency (int): Frequency of evaluation every n train steps
    eval_steps (int): Evaluation steps to be performed
  """
  def __init__(self, graph, examples, model, checkpoint_dir, display_fetches ):
      self.graph = graph
      self.examples = examples
      self.model = model
      self._checkpoint_dir = checkpoint_dir   
      self.output_dir  = checkpoint_dir
      self.display_fetches = display_fetches
      self._latest_checkpoint = None
      self.step =0
      # Saver class add ops to save and restore
      # variables to and from checkpoint
      with graph.as_default():
          self._saver = tf.train.Saver()
      self._eval_lock = threading.Lock()
      self._checkpoint_lock = threading.Lock()
      print("done setting up hook")
  def _update_latest_checkpoint(self):
      """Update the latest checkpoint file created in the output dir."""
      if self._checkpoint_lock.acquire(False):
          try:
              latest = tf.train.latest_checkpoint(self._checkpoint_dir)
              if not latest == self._latest_checkpoint:
                  self._latest_checkpoint = latest
          finally:
              self._checkpoint_lock.release()

  def after_run(self, run_context, run_values):
      print("hook after run")
      self.step = self.step +1
      if self.step%10 == 0:
          self._update_latest_checkpoint()
          if self._eval_lock.acquire(False):
              try:
                  self._run_eval()
              finally:
                  self._eval_lock.release()
  def end(self, session):
      """Called at then end of session to make sure we always evaluate."""
      self._update_latest_checkpoint()
      
      with self._eval_lock:
          self._run_eval()

  def _run_eval(self):
        with tf.Session(graph=self.graph) as session:
            self._saver.restore(session, self._latest_checkpoint)
             
            fetches = {
                "train": self.model.train,
                "global_step": self.model.global_step,
            }


#            fetches["display"] = self.display_fetches

            print("running eval session...")
            results = session.run(fetches)
            print("global step: "+str(results["global_step"]))

            #    sv.summary_writer.add_summary(results["summary"], results["global_step"])

            print("saving display images")
#            filesets = save_images(results["display"], self.output_dir, step=results["global_step"])
#            append_index(filesets, step=True)





  def end(self, session):
    """Called at then end of session to make sure we always evaluate."""
    print("hook end")

def add_stuff(examples, model):
    
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)


    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    # with tf.name_scope("predict_real_summary"):
    #    tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))


    # with tf.name_scope("predict_fake_summary"):
    #    tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    return display_fetches

def run(target, is_chief, job_name, a):
    input_dir = BASE_DIR+"/"+a.topic
    output_dir = BASE_DIR+"/"+a.topic+"/output"

    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)


    if is_chief:

        # evaluation_graph = tf.Graph()
        # with evaluation_graph.as_default():

        #     examples = load_examples(input_dir, "test", a.scale_size, a.batch_size)
        #     model = create_model(
        #         examples.inputs,
        #         examples.targets,
        #         a.num_generator_filters,
        #         a.num_discriminator_filters,
        #         a.gan_weight,
        #         a.l1_weight,
        #         a.lr,
        #         a.beta1)
        #     display_fetches = add_stuff(examples, model)


        #     hooks = [EvaluationRunHook(evaluation_graph, examples, model, output_dir, display_fetches)]
        hooks = []
    else:
        hooks = []

    graph=tf.Graph()
    with graph.as_default():
        # Placement of ops on devices using replica device setter
        # which automatically places the parameters on the `ps` server
        # and the `ops` on the workers
        #
        # See:
        # https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
        with tf.device(tf.train.replica_device_setter()):

            examples = load_examples(input_dir, "train", a.scale_size, a.batch_size)
            print("examples count = %d" % examples.count)

            # inputs and targets are [batch_size, height, width, channels]
            model = create_model(
                examples.inputs,
                examples.targets,
                a.num_generator_filters,
                a.num_discriminator_filters,
                a.gan_weight,
                a.l1_weight,
                a.lr,
                a.beta1)
            display_fetches = add_stuff(examples, model)
            saver = tf.train.Saver(max_to_keep=1)

            logdir = output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None

            with tf.train.MonitoredTrainingSession(master=target,
                                                   is_chief=is_chief,
                                                   checkpoint_dir=output_dir,
                                                   hooks=hooks,
                                                   save_checkpoint_secs=2000,
                                                   save_summaries_steps=50) as session:

                print("monitored session created.")
#            sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
#            with sv.managed_session() as sess:

                if a.checkpoint is not None:
                    print("loading model from checkpoint")
                    checkpoint = tf.train.latest_checkpoint(a.checkpoint)
                    saver.restore(session, checkpoint)

                max_steps = 2**30
                if a.max_epochs is not None:
                    max_steps = examples.steps_per_epoch * a.max_epochs
                if a.max_steps is not None:
                    max_steps = a.max_steps


                # training
                start = time.time()
                print("here we go")

                for step in range(max_steps):
                    print("step: " + str(step))
                    def should(freq):
                        return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                    options = None
                    run_metadata = None
                    if should(a.trace_freq):
                        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                    fetches = {
                        "train": model.train,
                        "global_step": model.global_step,
                    }

                    if is_chief:
                        if should(a.progress_freq):
                            fetches["discrim_loss"] = model.discrim_loss
                            fetches["gen_loss_GAN"] = model.gen_loss_GAN
                            fetches["gen_loss_L1"] = model.gen_loss_L1

                        if should(a.summary_freq):
                            #                            fetches["summary"] = sv.summary_op
                            print("should summary")
                        if should(a.display_freq):
                            fetches["display"] = display_fetches
                    try:        
                        results = session.run(fetches, options=options, run_metadata=run_metadata)
                        print("global step: "+str(results["global_step"]))
                        if is_chief:

                            if should(a.summary_freq):
                                print("recording summary")
                            #    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                            if should(a.display_freq):
                                print("saving display images")
                                filesets = save_images(results["display"], output_dir, step=results["global_step"])
                                append_index(filesets, step=True)

                            if should(a.trace_freq):
                                print("recording trace")
                             #   sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                            if should(a.progress_freq):
                                # global_step will have the correct step count if we resume from a checkpoint
                                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                                rate = (step + 1) * a.batch_size / (time.time() - start)
                                remaining = (max_steps - step) * a.batch_size / rate
                                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                                print("discrim_loss", results["discrim_loss"])
                                print("gen_loss_GAN", results["gen_loss_GAN"])
                                print("gen_loss_L1", results["gen_loss_L1"])

                            if should(a.save_freq):
                                print("saving model")
                                #                            saver.save(session, os.path.join(output_dir, "model"), global_step=sv.global_step)

                    except:
                        print("caught exeption")
                    if session.should_stop():
                        break




                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", help="e.g. simpson. should match folder name in GCS")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
    parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
    parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="save model every save_freq steps, 0 to disable")

    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
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



