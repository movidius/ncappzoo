from movidius.train import TrainConfig, do_train
from .model import mobilenet_v2_cifar10
from ..preprocessing import preprocess_train_cifar10, preprocess_test_cifar10
import tensorflow as tf


def train_cifar10(work_dir, epochs, reset_optim, phase, **kwargs):
    cfg = TrainConfig()
    cfg.build_fn = mobilenet_v2_cifar10
    cfg.loss_fn = tf.losses.softmax_cross_entropy
    cfg.img_shape = (32, 32)
    cfg.batch_size = 128
    cfg.train_split = 'cifar-10-train'
    cfg.eval_split = 'cifar-10-eval'
    cfg.work_dir = work_dir
    cfg.epochs = epochs
    cfg.fine_tune_var_list_fn = None
    cfg.optimizer = 'sgd'
    cfg.learning_rate = 0.1
    cfg.momentum = 0.9
    cfg.pretrained_ckpt = None
    cfg.reset_optim = reset_optim
    cfg.preprocess_train_fn = preprocess_train_cifar10
    cfg.preprocess_eval_fn = preprocess_test_cifar10

    do_train(cfg)
