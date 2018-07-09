from pathlib import Path

from .common import build_fn, train_var_list_fn
import tensorflow as tf
from functools import partial

from movidius.train import TrainConfig, train

INCEPTION_V4_CHECKPOINT = Path('checkpoints/inception_v4/inception_v4.ckpt')


def pretrained_var_list_fn():
    var_list = tf.global_variables('InceptionV4')
    var_list = [x for x in var_list if not x.name.startswith('InceptionV4/Logits')]
    return var_list


def run(clean, work_dir, epochs, train_split, eval_split):
    cfg = TrainConfig()
    cfg.build_fn = partial(build_fn, is_training=True)
    cfg.train_var_list_fn = train_var_list_fn
    cfg.pretrained_var_list_fn = pretrained_var_list_fn
    cfg.pretrained_ckpt = str(INCEPTION_V4_CHECKPOINT)
    cfg.loss_fn = tf.losses.softmax_cross_entropy
    cfg.img_shape = (299, 299)
    cfg.batch_size = 32
    cfg.train_split = train_split
    cfg.eval_split = eval_split
    cfg.work_dir = work_dir
    cfg.epochs = epochs
    cfg.clean = clean
    # cfg.train_variable_scopes = [
    #     'InceptionV4/Mixed_7a', 'InceptionV4/Mixed_7b', 'InceptionV4/Mixed_7c',
    #     'InceptionV4/Mixed_7d', 'InceptionV4/Logits'
    # ]
    cfg.fine_tune_var_list_fn = None  # train on full model

    train(cfg)
