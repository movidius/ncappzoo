import tensorflow.contrib.slim as slim
from .mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
import tensorflow as tf

SCOPE = 'MobilenetV1'
SHAPE = (224, 224)
SHAPE_v2 = (299, 299)


def build_fn(is_training, images, num_classes, depth_multiplier=1.0):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        logits, endpoints = mobilenet_v1(inputs=images, num_classes=num_classes, is_training=is_training,
                                         depth_multiplier=depth_multiplier, global_pool=False)
        # Compensate network tendency to overshoot logits to extremes
        # proba = tf.add(tf.nn.softmax(logits), 0.05, name='proba')
        proba = tf.nn.softmax(logits, name='proba')
        return logits, proba


def build_fn_v2(is_training, images, num_classes):
    return build_fn(is_training, images, num_classes, 1.5)


def model_var_list():
    var_list = tf.global_variables(SCOPE)
    return var_list
