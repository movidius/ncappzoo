import tensorflow.contrib.slim as slim
import tensorflow as tf
from .inception_v3 import inception_v3, inception_v3_arg_scope
from . import SHAPE

SCOPE = 'InceptionV3'


def create_build_fn(create_aux_logits=None, final_endpoint='Mixed_7c'):
    def build_fn(is_training, images, num_classes):
        _create_aux_logits = create_aux_logits if create_aux_logits is not None else is_training
        with slim.arg_scope(inception_v3_arg_scope(batch_norm_decay=0.95, updates_collections=None)):
            logits, endpoints = inception_v3(inputs=images, num_classes=num_classes, is_training=is_training,
                                             create_aux_logits=_create_aux_logits, final_endpoint=final_endpoint)
            # Compensate network tendency to overshoot logits to extremes
            # proba = tf.add(tf.nn.softmax(logits), 0.005, name='proba')
            proba = tf.nn.softmax(logits, name='proba')

            if _create_aux_logits:
                logits = (logits, endpoints['AuxLogits'])

            return logits, proba

    return build_fn


def loss_fn(onehot_labels, logits, scope, **kwargs):
    if isinstance(logits, tuple):
        main_loss = tf.losses.softmax_cross_entropy(onehot_labels, logits[0], scope=f'{scope}/main_loss')
        aux_loss = tf.losses.softmax_cross_entropy(onehot_labels, logits[1], scope=f'{scope}/aux_loss')
        loss = tf.add(main_loss, aux_loss, name=f'{scope}/loss')
    else:
        loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, scope=f'{scope}/loss')
    return loss


def create_pretrained_var_list_fn(logits=False):
    def fn():
        var_list = tf.global_variables('InceptionV3')
        var_list = [
            v for v in var_list if 'InceptionV3/Logits' not in v.name
        ]
        return var_list

    return fn
