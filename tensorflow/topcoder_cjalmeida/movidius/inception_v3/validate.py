from pathlib import Path

from movidius.validate import ValidateConfig, validate
from .inception_v3 import inception_v3, inception_v3_arg_scope
import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_fn(images, num_classes):
    with slim.arg_scope(inception_v3_arg_scope()):
        logits, endpoints = inception_v3(inputs=images, num_classes=num_classes, spatial_squeeze=False,
                                         is_training=False)
        proba = tf.nn.softmax(logits, name='proba')
        return logits, proba


def run():
    cfg = ValidateConfig()
    cfg.img_shape = (299, 299)
    cfg.eval_split = 'imagenet_val'
    cfg.batch_size = 64
    cfg.build_fn = build_fn
    cfg.checkpoint_file = Path('checkpoints/inception_v3/inception_v3.ckpt')
    validate(cfg)
