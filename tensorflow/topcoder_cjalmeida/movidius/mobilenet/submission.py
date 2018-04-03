from pathlib import Path
from functools import partial

from movidius.preprocessing import preprocess_eval
from movidius.submission import SubmissionConfig, do_submit
import tensorflow as tf
import numpy as np
from .train import build_finetune_net, finish_net_fn

SHAPE = (224, 224)


def build_fn(images, num_classes):
    net, _ = build_finetune_net(is_training=False)(images)
    logits, proba = finish_net_fn(net, num_classes=num_classes, is_training=False)


def preprocess_fn(img, meta):
    return np.expand_dims(preprocess_eval(img, meta, *SHAPE), 0)


def submit(work_dir: Path, step: int, skip_compile: bool, skip_inference: bool, score: bool, num: int, test_split: str,
           limit: int, **kwargs):
    cfg = SubmissionConfig()
    cfg.model_name = 'MobileNet V1'
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
