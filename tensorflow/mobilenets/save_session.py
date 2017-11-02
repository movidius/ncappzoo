#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 


import numpy as np
import tensorflow as tf
import sys

#from tensorflow.contrib.slim.nets.mobilenet_v1 import *
from mobilenet_v1 import *

IMGCLASS = "1.0"
IMGSIZE = "224"

if len(sys.argv) > 1:
    IMGCLASS = sys.argv[1]
    if len(sys.argv) > 2:
      IMGSIZE = sys.argv[2]
else:
    print("WARNING: using", IMGCLASS, " for MobileNet Class")
    print("WARNING: using", IMGSIZE, " for MobileNet image size")
    print("Run with save_session.py [IMGCLASS] [IMGSIZE] to change")
    input("Press enter to continue")

IMGSIZE_HACK = str(int(IMGSIZE) - 1)

MOBILENETNAME="mobilenet_v1_" + IMGCLASS + "_" + IMGSIZE
MOBILENETNAME_HACK="mobilenet_v1_" + IMGCLASS + "_" + IMGSIZE_HACK

print("Saving Session for ", MOBILENETNAME, " HACK ", MOBILENETNAME_HACK)

slim = tf.contrib.slim

def run(name, image_size, num_classes):
  with tf.Graph().as_default():
    image = tf.placeholder("float", [1, image_size, image_size, 3], name="input")
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training = False)):
        if IMGCLASS == "0.25":
            logits, end_points = mobilenet_v1_025(image, num_classes, is_training=False)
        elif IMGCLASS == "0.50": 
            logits, end_points = mobilenet_v1_050(image, num_classes, is_training=False)
        elif IMGCLASS == "0.75": 
            logits, end_points = mobilenet_v1_075(image, num_classes, is_training=False)
        else:
            logits, end_points = mobilenet_v1(image, num_classes, is_training=False)
    probabilities = tf.identity(end_points['Predictions'], name='output')
    init_fn = slim.assign_from_checkpoint_fn(MOBILENETNAME + '.ckpt', slim.get_model_variables('MobilenetV1'))

    with tf.Session() as sess:
        init_fn(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, "output/"+name)

run(MOBILENETNAME,      IMGSIZE,      1001)
run(MOBILENETNAME_HACK, IMGSIZE_HACK, 1001)
