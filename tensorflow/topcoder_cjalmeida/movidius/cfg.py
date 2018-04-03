from pathlib import Path

NASNET_MOBILE_CHECKPOINT = Path('checkpoints/nasnet_mobile/model.ckpt')
NASNET_MOBILE_META = Path('checkpoints/nasnet_mobile/model.meta')

INPUT = Path('input')
INPUT_PROVISIONAL = INPUT / 'provisional'
INPUT_TRAINING = INPUT / 'training'
CATEGORIES = INPUT / 'categories.txt'
INPUT_TRAINING_LABELS = INPUT / 'training_ground_truth.csv'

IMAGENET_CAT_1001 = INPUT / 'cat_1001.txt'
IMAGENET = Path('/mnt/ds/cjalmeida/dataset/Imagenet')
IMAGENET_2012_VAL = IMAGENET / '2012/validation.h5'
IMAGENET_2012_VAL_DEVKIT = IMAGENET / '2012/devkit'
IMAGENET_EXTRA_DIR = INPUT / 'extra_images'
IMAGENET_URL_FILES = IMAGENET / 'fall11_urls.txt'
IMAGENET_STRUCTURE_XML = IMAGENET / 'structure_released.xml'
NCS_NCCOMPILE = 'mvNCCompile'
NCS_NCCHECK = '/usr/local/bin/mvNCCheck'
NCS_NCPROFILE = 'sdk/NCSDK/ncsdk-x86_64/tk/mvNCProfile.py'

CIFAR = Path('/mnt/ds/cjalmeida/dataset/CIFAR')
CIFAR_10_DIR = CIFAR / 'cifar-10-batches-py'

SEED = 41


def set_seed():
    import tensorflow as tf
    import numpy as np
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
