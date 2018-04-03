from collections import namedtuple, Counter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import pickle
import re
import h5py

from tqdm import tqdm

from movidius.dataset.utils import md5sum

ImageNetMeta = namedtuple('ImageNetMeta', 'key file label wnid')


def convert_to_hdf5(images_path: Path, out_file: Path):
    f = h5py.File(str(out_file), mode='w')
    files = list(images_path.glob('*.JPEG'))
    code_re = re.compile(r'.*_val_(\d+).JPEG')
    files.sort(key=lambda x: int(code_re.match(x.name).group(1)))
    for file in tqdm(files):  # type: Path
        data = np.frombuffer(file.read_bytes(), 'u1')
        f.create_dataset(file.name, data=data)
    f.close()


class ImagenetExtrasDataset:
    def __init__(self, extras_path: Path, wnid_to_label: Dict, only_mapped_synsets=True):
        self.extras_path = extras_path
        self.wnid_to_label = wnid_to_label
        self.only_mapped_synsets = only_mapped_synsets
        self._load_meta()

    def _load_meta(self):
        hash_key = md5sum([str(self.extras_path), str(self.wnid_to_label), str(self.only_mapped_synsets)])
        cache = f'./work/Extras_{hash_key}.pkl'
        try:
            self.meta = pickle.load(open(cache, 'rb'))
        except FileNotFoundError:
            self.meta: List[ImageNetMeta] = []

            for f in self.extras_path.iterdir():  # type: Path
                if f.suffix.lower() in ('.jpeg', '.jpg'):
                    key = f.name
                    wnid, _ = f.with_suffix('').name.split('_', maxsplit=1)
                    label = self.wnid_to_label.get(wnid)
                    if self.only_mapped_synsets and not label:
                        continue
                    self.meta.append(ImageNetMeta(key, f, label, wnid))

            with open(cache, 'wb') as cache_file:
                pickle.dump(self.meta, cache_file)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Tuple[np.ndarray, ImageNetMeta]:
        while True:
            try:
                meta = self.meta[index]
                img = cv2.imread(str(meta.file), cv2.IMREAD_COLOR)
                if img is None:
                    raise Exception
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img, meta
            except:
                index += 1 if index < len(self.meta) else 0
                pass


class ILSVRCDataset:
    def __init__(self, images_path: Path, devkit_path: Path = None, wnid_map='inception') -> None:
        self.images_path = images_path
        self.devkit_path = devkit_path
        self.wnid_map = 'inception'
        self.meta: List[ImageNetMeta] = []
        self.code_re = re.compile(r'.*_val_(\d+).JPEG')
        self.h5 = True if images_path.name.endswith('.h5') else None
        self.synsets = {}
        self.label_to_wnid = {}
        self._h5f: h5py.File = None
        self._load_meta()

    def _load_meta(self):
        hash_key = md5sum([str(self.images_path), str(self.devkit_path), str(self.wnid_map)])
        cache = f'./work/ILSVRC_{hash_key}.pkl'
        try:
            self.meta, self.synsets, self.label_to_wnid = pickle.load(open(cache, 'rb'))
        except FileNotFoundError:
            from scipy.io import loadmat
            wnid_map = self._load_wnid_map()
            self.label_to_wnid = {(v + 1): k for k, v in wnid_map.items()}
            i_meta_file = self.devkit_path / 'data' / 'meta.mat'
            i_meta = loadmat(str(i_meta_file))['synsets']
            truth_file = self.devkit_path / 'data' / 'ILSVRC2012_validation_ground_truth.txt'
            codes = [int(x) - 1 for x in truth_file.read_text().splitlines()]
            self.synsets = {i_meta[i][0][1][0]: i_meta[i][0][2][0] for i in set(codes)}
            wnids = [i_meta[i][0][1][0] for i in codes]
            labels = [wnid_map[i] + 1 for i in wnids]
            if self.h5:
                h5f = h5py.File(str(self.images_path), 'r')
                keys = [(None, k) for k in h5f.keys()]
                h5f.close()
            else:
                files = list(self.images_path.glob('*.JPEG'))
                keys = [(_file, _file.name) for _file in files]

            print('Loading ILSVRC metadata')
            for _file, key in tqdm(keys):
                code = int(self.code_re.match(key).group(1)) - 1
                label = labels[code]
                wnid = wnids[code]
                self.meta.append(ImageNetMeta(key, str(_file), label, wnid))
            with open(cache, 'wb') as f:
                pickle.dump((self.meta, self.synsets, self.label_to_wnid), f)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Tuple[np.ndarray, ImageNetMeta]:
        meta = self.meta[index]
        if self.h5 and not self._h5f:
            self._h5f = h5py.File(str(self.images_path), 'r')  # lazy load File due to multiprocessing

        if self._h5f:
            data: np.ndarray = self._h5f[meta.key][:]
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(meta.file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, meta

    def _load_wnid_map(self):
        if self.wnid_map != 'inception':
            return NotImplementedError
        synsets = Path('input/inception/imagenet_lsvrc_2015_synsets.txt').read_text().splitlines()
        synsets = {v: i for i, v in enumerate(synsets)}
        return synsets


if __name__ == '__main__':
    from movidius import cfg

    convert_to_hdf5(cfg.IMAGENET_2012_VAL, cfg.IMAGENET_2012_VAL.with_suffix('.h5'))
