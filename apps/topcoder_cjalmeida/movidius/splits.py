from typing import Tuple, Any

from movidius.cfg import IMAGENET_2012_VAL, IMAGENET_2012_VAL_DEVKIT, INPUT_TRAINING, INPUT_TRAINING_LABELS, \
    INPUT_PROVISIONAL, IMAGENET_EXTRA_DIR, CIFAR_10_DIR
from movidius.dataset.cifar import Cifar10Dataset
from movidius.dataset.movidius import MovidiusChallengeDataset
from movidius.dataset.imagenet import ILSVRCDataset, ImagenetExtrasDataset
from movidius.dataset.utils import MergeDataset


def dataset_from_split(split) -> Tuple[Any, int]:
    if split == 'imagenet_val':
        return ILSVRCDataset(IMAGENET_2012_VAL, IMAGENET_2012_VAL_DEVKIT), 1001
    elif split in ('train', 'eval', 'small', 'all'):
        return MovidiusChallengeDataset(INPUT_TRAINING, split=split, labels_file=INPUT_TRAINING_LABELS), 201
    elif split in ('test', 'test-small'):
        return MovidiusChallengeDataset(INPUT_PROVISIONAL, split=split), 201
    elif split == 'train-extra':
        train_ds, n = dataset_from_split('train')
        extras_ds = ImagenetExtrasDataset(IMAGENET_EXTRA_DIR, wnid_to_label=train_ds.wnid_to_label)
        return MergeDataset([train_ds, extras_ds]), n
    elif split == 'cifar-10-train':
        return Cifar10Dataset(CIFAR_10_DIR, 'train'), 10
    elif split == 'cifar-10-eval':
        return Cifar10Dataset(CIFAR_10_DIR, 'eval'), 10
    else:
        raise NotImplementedError(('split', split))
