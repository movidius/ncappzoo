# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a TFRECORD path"""

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from preprocessing import preprocessing_factory
from nets import nets_factory
import time
import os
import math
from data import get_split, load_batch
slim = tf.contrib.slim

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
tf.app.flags.DEFINE_string('checkpoint_path', None, 'direcotry where to find model files')
tf.app.flags.DEFINE_string('eval_dir', './evallog', 'direcotry where to create log files')
tf.app.flags.DEFINE_string('dataset_dir', '/home/local/TECHNICALABS/alu/data/falldetect/data/processed/tfrecord/', 'directory to where Validation TFRecord files are')
tf.app.flags.DEFINE_integer('num_classes', 2, 'number of classes')
tf.app.flags.DEFINE_string('file_pattern', 'falldata_%s_*.tfrecord', 'file pattern of TFRecord files')
tf.app.flags.DEFINE_string('file_pattern_for_counting', 'falldata', 'identify tfrecord files')
tf.app.flags.DEFINE_string('labels_file', None, 'path to labels file')
tf.app.flags.DEFINE_integer('image_size', None, 'image size ISxIS')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1', 'name of model architecture defined in nets factory')
tf.app.flags.DEFINE_string('preprocessing_name', 'lenet', 'name of model preprocessing defined in preprocessing factory')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'number of epochs to evaluate for')
tf.app.flags.DEFINE_float( 'moving_average_decay', None, 'The decay to use for the moving average.' 'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_integer('max_num_batches', None, 'Max number of batches to evaluate by default use all.')
FLAGS = tf.app.flags.FLAGS

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.checkpoint_path:
    raise ValueError('You must supply the checkpoint directory with --checkpoint_path')
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = get_split('validation', FLAGS.dataset_dir, FLAGS.num_classes, FLAGS.labels_file, FLAGS.file_pattern, FLAGS.file_pattern_for_counting)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,shuffle=False,common_queue_capacity=2 * FLAGS.batch_size,common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=FLAGS.num_classes, is_training=False)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)
    eval_image_size = FLAGS.image_size or network_fn.default_image_size
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    images, labels = tf.train.batch([image, label],batch_size=FLAGS.batch_size, num_threads=FLAGS.num_preprocessing_threads, capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)
    variables_to_restore = slim.get_variables_to_restore()
    predictions = tf.argmax(logits, 1)
    labels = tf.reshape(tf.squeeze(labels), (FLAGS.batch_size, ))
    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
                                                                            'Recall_5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)

if __name__ == '__main__':
    tf.app.run()