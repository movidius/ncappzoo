import tensorflow.contrib.slim as slim
from .inception_v4 import inception_v4, inception_v4_arg_scope
import tensorflow as tf

SCOPE = 'InceptionV4'
SHAPE = (299, 299)


def build_fn(is_training, images, num_classes):
    with slim.arg_scope(inception_v4_arg_scope()):
        logits, endpoints = inception_v4(inputs=images, num_classes=num_classes, is_training=is_training,
                                         create_aux_logits=False)
        # Compensate network tendency to overshoot logits to extremes
        proba = tf.add(tf.nn.softmax(logits), 0.05, name='proba')
        return logits, proba


def train_var_list_fn():
    var_list = tf.global_variables('InceptionV4')
    return var_list
