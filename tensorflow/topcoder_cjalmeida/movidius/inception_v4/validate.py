from pathlib import Path

from movidius.validate import ValidateConfig, validate
from .inception_v4 import inception_v4, inception_v4_arg_scope
import tensorflow as tf
import tensorflow.contrib.slim as slim
from .common import build_fn

def run():
    cfg = ValidateConfig()
    cfg.img_shape = (299, 299)
    cfg.eval_split = 'imagenet_val'
    cfg.batch_size = 64
    cfg.build_fn = build_fn
    cfg.checkpoint_file = Path('checkpoints/inception_v4/inception_v4.ckpt')
    validate(cfg)
