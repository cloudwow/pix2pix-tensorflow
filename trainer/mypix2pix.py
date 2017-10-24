
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

CROP_SIZE = 512


def add_stuff(examples, model):


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

def run(target, is_chief, job_name, a):
    input_dir = BASE_DIR+"/"+a.topic
    if a.output_dir==None:
        output_dir = BASE_DIR+"/"+a.topic+"/output"
    else:
        output_dir  = a.output_dir

    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)


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
            print(a.training_image_dir)
            print(type(a.training_image_dir))

            training_image_dirs =  []
            for dirname in a.training_image_dir:
                print(dirname)
                if dirname.startswith("/") or dirname.startswith("./"):
                     training_image_dirs.append( dirname)
                else :
                    training_image_dirs.append( input_dir+"/" + dirname)
            print(training_image_dirs)
            import myp2p.model as p2p_model

            examples =p2p_model.load_examples(training_image_dirs, a.scale_size, a.batch_size)
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
            add_stuff(examples, model)
            saver = tf.train.Saver(max_to_keep=1)

            logdir = output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None

            with tf.train.MonitoredTrainingSession(master=target,
                                                   is_chief=is_chief,
                                                   checkpoint_dir=output_dir,
                                                   hooks=hooks,
                                                   save_checkpoint_secs=1800,
                                                   save_summaries_steps=1000) as session:

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
                    try:        
                        results = session.run(fetches, options=options, run_metadata=run_metadata)
                        print("global step: "+str(results["global_step"]))
                        if is_chief:

                            if should(a.summary_freq):
                                print("recording summary")
                            #    sv.summary_writer.add_summary(results["summary"], results["global_step"])


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
    parser.add_argument("--training_image_dir", action="append")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=1000, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=500, help="display progress every progress_freq steps")
    parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="save model every save_freq steps, 0 to disable")

    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
    parser.add_argument("--num_generator_filters", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--num_discriminator_filters", type=int, default=64, help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=512, help="scale images to this size before cropping to 512x512")
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



