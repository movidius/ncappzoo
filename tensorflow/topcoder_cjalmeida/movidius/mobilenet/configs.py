from functools import partial
from pathlib import Path
from movidius.profile import ProfileConfig, do_profile
from movidius.train import TrainConfig, do_train
from movidius.validate import ValidateConfig, do_validate
from .common import SHAPE, build_fn, SHAPE_v2, build_fn_v2
from ..preprocessing import preprocess_test_cifar10, preprocess_train_cifar10
import tensorflow as tf


def profile(**kwargs):
    cfg = ProfileConfig()
    cfg.shape = SHAPE
    cfg.build_fn = build_fn
    cfg.temp_dir = kwargs.get('outdir')
    do_profile(cfg)


def profile_v2(**kwargs):
    cfg = ProfileConfig()
    cfg.shape = SHAPE_v2
    cfg.build_fn = build_fn_v2
    cfg.temp_dir = kwargs.get('outdir')
    do_profile(cfg)


def train_v2(train_split, eval_split, work_dir, epochs, **kwargs):
    cfg = TrainConfig()
    cfg.build_fn = build_fn_v2
    cfg.loss_fn = tf.losses.softmax_cross_entropy
    cfg.img_shape = SHAPE_v2
    cfg.batch_size = 32
    cfg.train_split = train_split
    cfg.eval_split = eval_split
    cfg.work_dir = work_dir
    cfg.epochs = epochs
    cfg.fine_tune_var_list_fn = None
    cfg.optimizer = 'adam'
    cfg.learning_rate = 0.001
    cfg.pretrained_ckpt = None
    do_train(cfg)


def train(train_split, eval_split, work_dir, epochs, **kwargs):
    cfg = TrainConfig()
    cfg.build_fn = build_fn
    cfg.loss_fn = tf.losses.softmax_cross_entropy
    cfg.img_shape = SHAPE
    cfg.batch_size = 32
    cfg.train_split = train_split
    cfg.eval_split = eval_split
    cfg.work_dir = work_dir
    cfg.epochs = epochs
    cfg.fine_tune_var_list_fn = None
    cfg.optimizer = 'adam'
    cfg.learning_rate = 0.001
    cfg.pretrained_ckpt = None
    do_train(cfg)


def train_cifar10(work_dir, epochs, **kwargs):
    cfg = TrainConfig()
    cfg.build_fn = build_fn
    cfg.loss_fn = tf.losses.softmax_cross_entropy
    cfg.img_shape = (32, 32)
    cfg.batch_size = 128
    cfg.train_split = 'cifar-10-train'
    cfg.eval_split = 'cifar-10-eval'
    cfg.work_dir = work_dir
    cfg.epochs = epochs
    cfg.fine_tune_var_list_fn = None
    cfg.optimizer = 'adam'
    cfg.learning_rate = 0.01
    # cfg.momentum = 0.
    cfg.pretrained_ckpt = None
    cfg.preprocess_train_fn = preprocess_train_cifar10
    cfg.preprocess_eval_fn = preprocess_test_cifar10
    do_train(cfg)


def train_cifar10_v2(work_dir, epochs, **kwargs):
    cfg = TrainConfig()
    cfg.build_fn = build_fn_v2
    cfg.loss_fn = tf.losses.softmax_cross_entropy
    cfg.img_shape = (48, 48)
    cfg.batch_size = 64
    cfg.train_split = 'cifar-10-train'
    cfg.eval_split = 'cifar-10-eval'
    cfg.work_dir = work_dir
    cfg.epochs = epochs
    cfg.fine_tune_var_list_fn = None
    cfg.optimizer = 'adam'
    cfg.learning_rate = 0.001
    cfg.pretrained_ckpt = None
    do_train(cfg)


def validate_v2(work_dir: Path, step: int, **kwargs):
    cfg = ValidateConfig()
    cfg.img_shape = SHAPE_v2
    cfg.eval_split = 'train'
    cfg.batch_size = 32
    cfg.build_fn = build_fn_v2
    cfg.checkpoint_file = work_dir / 'save' / f'model-{step}'
    do_validate(cfg)


def validate_imagenet():
    cfg = ValidateConfig()
    cfg.img_shape = SHAPE
    cfg.eval_split = 'imagenet_val'
    cfg.batch_size = 32
    cfg.build_fn = build_fn
    cfg.checkpoint_file = Path('checkpoints/mobilenet_100_128/mobilenet_v1_1.0_128.ckpt')
    do_validate(cfg)
