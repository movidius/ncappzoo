import pickle

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from tqdm import tqdm

from ..cfg import SEED
from .imagenet import ImageNetMeta
from .utils import md5sum

EVAL_SPLIT_SIZE = 0.1


class MovidiusChallengeDataset:
    def __init__(self, base_dir: Path, split: str = 'all', labels_file: Path = None) -> None:
        self.base_dir = base_dir
        self.labels_file = labels_file
        self.split = split

        self.meta: List[ImageNetMeta] = []
        self.wnid_to_label: Dict[str, int] = dict()
        self._load_meta()

    def _load_meta(self):
        assert self.split in ('train', 'eval', 'small', 'all', 'test', 'test-small')

        hash_key = md5sum([str(self.base_dir), str(self.labels_file), str(self.split)])
        cache = f'./work/movds_{hash_key}.pkl'
        try:
            (self.meta, self.wnid_to_label) = pickle.load(open(cache, 'rb'))
        except FileNotFoundError:
            labels = None
            if self.labels_file:
                labels = [x.split(',') for x in self.labels_file.read_text().splitlines()]
                self.wnid_to_label = {x[2]: int(x[1]) for x in labels[1:]}
                labels = {x[0]: (int(x[1]), x[2]) for x in labels[1:]}
            files = list(self.base_dir.iterdir())
            print('Loading Imagenet metadata')
            for f in tqdm(files):
                if not f.name.endswith('.jpg'):
                    continue
                key = f.name
                label, wnid = labels[key] if labels is not None else (None, None)
                self.meta.append(ImageNetMeta(key, str(f), label, wnid))
            self._apply_split()
            with open(cache, 'wb') as f:
                pickle.dump((self.meta, self.wnid_to_label), f)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Tuple[np.ndarray, ImageNetMeta]:
        meta = self.meta[index]
        img = cv2.imread(str(meta.file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, meta

    def _apply_split(self):
        """
        Split training data between train/eval sets with balanced classes. We rely on properly set np.random.seed
        """
        if self.split is None or self.split in ('all', 'test'):
            return

        random = np.random.RandomState(SEED)
        random.shuffle(self.meta)
        if self.split == 'eval':
            p = EVAL_SPLIT_SIZE
        elif self.split in ('small', 'test-small'):
            p = 0.05
            self.meta.reverse()
        elif self.split == 'train':
            p = 1 - EVAL_SPLIT_SIZE
            self.meta.reverse()
        else:
            raise NotImplementedError(('split', self.split))

        new_meta = []
        grouped = {}
        for item in self.meta:  # type: ImageNetMeta
            grouped.setdefault(item.label, []).append(item)

        for group, items in grouped.items():
            n = int(round(len(items) * p))
            new_meta += items[:n]

        self.meta = new_meta
