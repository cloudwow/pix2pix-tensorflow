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
import myp2p.model as p2p_model

BASE_DIR="gs://pix2pixdata"

CROP_SIZE = 256

def save_images(fetches, output_dir, run_id):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + "_"+run_id+".png"
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

def start_index_row(output_dir):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table>\n\n")
    index.write("<tr>\n")
def end_index_row(output_dir):
    index_path = os.path.join(output_dir, "index.html")
    index = open(index_path, "a")
    index.write("</tr>\n\n")

def append_index(filesets, output_dir):
    index_path = os.path.join(output_dir, "index.html")
    index = open(index_path, "a")
    
    for fileset in filesets:

    
#        for kind in ["inputs", "outputs", "targets"]:
#            index.write("<td><img src='images/%s'></td>" % fileset[kind])
        index.write("    <td><img src='images/%s'></td>\n" % fileset["outputs"])

    return index_path
def add_stuff(examples, model):
    
    inputs = p2p_model.deprocess(examples.inputs)
    targets = p2p_model.deprocess(examples.targets)
    outputs = p2p_model.deprocess(model.outputs)


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

    # testing
    # at most, process the test data once

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

            examples = p2p_model.load_examples([input_dir+"/test_images"], a.scale_size, a.batch_size)
            print("examples count = %d" % examples.count)

            # inputs and targets are [batch_size, height, width, channels]
            model = p2p_model.create_model(
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
                max_steps = examples.steps_per_epoch
                import datetime
                run_id= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # training
                start = time.time()
                print("here we go")
                start_index_row(output_dir)

                for step in range(max_steps):
                    results = session.run(display_fetches)
                    filesets = save_images(results, output_dir, run_id)
                    for i, f in enumerate(filesets):
                        print("evaluated image", f["name"])
                        index_path = append_index(filesets, output_dir)

                        print("wrote index at", index_path)

                
                    if session.should_stop():
                        break
                end_index_row(output_dir)




                
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
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

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



