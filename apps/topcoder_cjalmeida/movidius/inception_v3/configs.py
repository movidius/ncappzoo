from pathlib import Path

from movidius.submission import SubmissionConfig, do_submit, ScoreConfig, do_score_gpu
from movidius.train import TrainConfig
import tensorflow as tf

from .common import create_build_fn, loss_fn, create_pretrained_var_list_fn, SHAPE
from ..preprocessing import preprocess_train_cifar10, preprocess_test_cifar10, preprocess_train, preprocess_eval
from ..train import do_train


def train_cifar10(epochs, work_dir, **kwargs):
    cfg = TrainConfig()

    cfg.build_fn = create_build_fn(create_aux_logits=False, final_endpoint='Mixed_5d')
    cfg.loss_fn = loss_fn
    cfg.preprocess_train_fn = preprocess_train_cifar10
    cfg.preprocess_eval_fn = preprocess_test_cifar10

    # cifar10 sets
    cfg.train_split = 'cifar-10-train'
    cfg.eval_split = 'cifar-10-eval'
    cfg.img_shape = (32, 32)
    cfg.batch_size = 64

    # optim
    cfg.optimizer = 'adam'
    cfg.momentum = 0.9
    cfg.learning_rate = 0.01

    # cmd params
    cfg.epochs = epochs
    cfg.work_dir = work_dir

    do_train(cfg)


def fine_tune_var_list_fn():
    var_list = []
    var_list += tf.trainable_variables('InceptionV3/Logits')
    var_list += tf.trainable_variables('InceptionV3/Mixed_7c')
    var_list += tf.trainable_variables('InceptionV3/Mixed_7b')
    var_list += tf.trainable_variables('InceptionV3/Mixed_7a')
    var_list = [v for v in var_list if 'BatchNorm' not in v.name]
    return var_list


def create_custom_initializer():
    var_list = []
    var_list += tf.trainable_variables('InceptionV3/Logits')
    var_list += tf.trainable_variables('InceptionV3/Mixed_7c')
    init = tf.initialize_variables(var_list)
    return init


def train_finetune(epochs, work_dir, phase, reset_optim, **kwargs):
    cfg = TrainConfig()

    # base movidius/imagenet conf
    cfg.build_fn = create_build_fn(create_aux_logits=False)
    cfg.force_is_training = False  # to force eval mode on batch_norm
    cfg.loss_fn = loss_fn
    cfg.preprocess_train_fn = preprocess_train
    cfg.preprocess_eval_fn = preprocess_eval
    cfg.train_split = 'train-extra'
    cfg.eval_split = 'eval'
    cfg.img_shape = SHAPE
    cfg.batch_size = 32

    # optim
    cfg.optimizer = 'adam'
    cfg.momentum = 0.9
    cfg.learning_rate = 0.001

    # pretrained
    cfg.pretrained_var_list_fn = create_pretrained_var_list_fn(logits=False)
    cfg.pretrained_ckpt = 'checkpoints/inception_v3/inception_v3.ckpt'
    cfg.fine_tune_var_list_fn = fine_tune_var_list_fn

    # cmd params
    cfg.epochs = epochs
    cfg.work_dir = work_dir

    cfg.reset_optim = reset_optim

    if phase == 2:
        cfg.optimizer = 'sgd'
        cfg.momentum = 0.9
        cfg.learning_rate = 0.001

    elif phase == 3:
        cfg.optimizer = 'sgd'
        cfg.momentum = 0.9
        cfg.learning_rate = 0.0001

    do_train(cfg)


def submit(work_dir: Path, step: int, skip_compile: bool, skip_inference: bool, score: bool, num: int, test_split: str,
           limit: int, skip_upload: bool, **kwargs):
    cfg = SubmissionConfig()
    cfg.model_name = 'Inception V3 (fine tuned)'
    cfg.weights_file = work_dir / 'save' / f'model-{step}'
    cfg.submission_dir = work_dir / 'submit'
    cfg.build_fn = create_build_fn(create_aux_logits=False)
    cfg.preprocess_fn = preprocess_eval
    cfg.num = num
    cfg.shape = SHAPE
    cfg.skip_compile = skip_compile
    cfg.skip_inference = skip_inference
    cfg.test_split = test_split
    cfg.score = score
    cfg.limit = limit
    cfg.skip_upload = skip_upload
    do_submit(cfg)


def score_gpu(work_dir: Path, step: int, split: str, limit: int, time: float, **kwargs):
    cfg = ScoreConfig()
    cfg.weights_file = work_dir / 'save' / f'model-{step}'
    cfg.build_fn = create_build_fn(create_aux_logits=False)
    cfg.preprocess_fn = preprocess_eval
    cfg.shape = SHAPE
    cfg.split = split
    cfg.limit = limit
    cfg.time = time
    do_score_gpu(cfg)