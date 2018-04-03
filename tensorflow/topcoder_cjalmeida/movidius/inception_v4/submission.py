from pathlib import Path
from functools import partial

from movidius.preprocessing import preprocess_eval
from movidius.submission import SubmissionConfig, do_submit, ScoreConfig, do_score_gpu
import tensorflow as tf
import numpy as np
from .common import build_fn

SHAPE = (299, 299)


def preprocess_fn(img, meta, *args, **kwargs):
    return np.expand_dims(preprocess_eval(img, meta, *SHAPE), 0)


def validate_imagenet():
    """ Pre-configured submission steps to validate ImageNet validation set accuracy on NCS device """
    work_dir = Path('/tmp/imagenet_eval/inception_v4')
    cfg = SubmissionConfig()
    cfg.shape = SHAPE
    cfg.model_name = 'Inception V4 (orig)'
    cfg.weights_file = 'checkpoints/inception_v4/inception_v4.ckpt'
    cfg.submission_dir = work_dir / 'submit'
    cfg.build_fn = build_fn
    cfg.preprocess_fn = preprocess_fn
    cfg.num = 1
    cfg.skip_compile = False
    cfg.skip_inference = False
    cfg.test_split = 'imagenet_val'
    cfg.score = True
    do_submit(cfg)


def score_gpu(work_dir: Path, step: int, split: str, limit: int, time: float, **kwargs):
    cfg = ScoreConfig()
    cfg.weights_file = work_dir / 'save' / f'model-{step}'
    cfg.build_fn = build_fn
    cfg.preprocess_fn = preprocess_eval
    cfg.shape = SHAPE
    cfg.split = split
    cfg.limit = limit
    cfg.time = time
    do_score_gpu(cfg)


def submit(work_dir: Path, step: int, skip_compile: bool, skip_inference: bool, score: bool, num: int, test_split: str,
           limit: int, **kwargs):
    cfg = SubmissionConfig()
    cfg.model_name = 'Inception V4'
    cfg.weights_file = work_dir / 'save' / f'model-{step}'
    cfg.submission_dir = work_dir / 'submit'
    cfg.build_fn = build_fn
    cfg.preprocess_fn = preprocess_fn
    cfg.num = num
    cfg.shape = SHAPE
    cfg.skip_compile = skip_compile
    cfg.skip_inference = skip_inference
    cfg.test_split = test_split
    cfg.score = score
    cfg.limit = limit
    do_submit(cfg)
