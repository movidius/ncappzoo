import tensorflow.contrib.slim as slim
from .nasnet import nasnet_mobile_arg_scope, build_nasnet_mobile
import tensorflow as tf

SHAPE = (224, 224)


def build_fn(is_training, images, num_classes):
    global _model_var_list
    with slim.arg_scope(nasnet_mobile_arg_scope()):
        logits, endpoints = build_nasnet_mobile(images=images, num_classes=num_classes, is_training=is_training,
                                                final_endpoint='Predictions')
        # Compensate network tendency to overshoot logits to extremes
        proba = tf.add(tf.nn.softmax(logits), 0.05, name='proba')
        return logits, proba


def model_var_list():
    var_list = tf.global_variables()
    return var_list
