# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from nets import nets_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')
tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')
tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')
tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of expected classes for output tensor')
tf.app.flags.DEFINE_string(
    'input_layer_name', 'input', 'Name of the input layer Placeholder tensor')
tf.app.flags.DEFINE_string(
    'output_layer_name', 'output', 'Name of the output layer softmax classification layer')
tf.app.flags.DEFINE_string(
    'ckpt_path', None, 'Name of the output layer softmax classification layer')
tf.app.flags.DEFINE_string(
    'output_ckpt_path', None, 'Name of output file path for checkpoint fused inference graph')
FLAGS = tf.app.flags.FLAGS

'''
Provide two different ways to export graph
1. Export graph structure only. Give FLAGS without ckpt_path/output_ckpt path. Exports a graphdef to use with the freeze graph tool to create a pb file
2. Export new checkpooint file. Give FLAGS with ckpt_path/output_ckpt path. Exports a new checkpoint file with an updated inference GraphDef
Tested both methods and have resulted in, in our tests, the same graph when converted via the Movidius Toolkit
'''
def main(_):
  if not (FLAGS.output_file or (FLAGS.output_ckpt_path and FLAGS.ckpt_path)):
    raise ValueError('Missing output file path OR missing ckpt path [{}] and output_ckpt_path [{}]'.format(FLAGS.ckpt_path, FLAGS.output_ckpt_path))
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=(FLAGS.num_classes - FLAGS.labels_offset), is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size
    placeholder = tf.placeholder("float", name=FLAGS.input_layer_name, shape=[FLAGS.batch_size, image_size, image_size, 3])
    logits, endpoints = network_fn(placeholder)
    #final_tensor = tf.nn.softmax(logits, name=FLAGS.output_layer_name)
    final_tensor = tf.identity(endpoints['Predictions'], name=FLAGS.output_layer_name)
    if FLAGS.ckpt_path:
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.ckpt_path, slim.get_variables_to_restore())
        with tf.Session() as sess:
            init_fn(sess)
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, FLAGS.output_ckpt_path)
    else:
        graph_def = graph.as_graph_def()
        with gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())

if __name__ == '__main__':
  tf.app.run()
