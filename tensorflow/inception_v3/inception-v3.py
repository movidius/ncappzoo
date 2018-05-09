#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 

import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

def run(name, image_size, num_classes):
  with tf.Graph().as_default():
    image = tf.placeholder("float", [1, image_size, image_size, 3], name="input")
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(image, num_classes, is_training=False, spatial_squeeze=False)
    probabilities = tf.nn.softmax(logits)
    init_fn = slim.assign_from_checkpoint_fn('inception_v3.ckpt', slim.get_model_variables('InceptionV3'))

    with tf.Session() as sess:
        init_fn(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, "output/"+name)

run('inception-v3', 299, 1001)
