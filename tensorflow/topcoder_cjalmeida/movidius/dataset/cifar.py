import pickle
import numpy as np
from pathlib import Path
from .imagenet import ImageNetMeta
import cv2


class Cifar10Dataset:
    test_size = (32, 32)

    def __init__(self, cifar10_dir: Path, split: str):
        self.meta = []
        if split == 'train':
            self.data = np.empty([50000, 32, 32, 3], 'u1')
            for batch in [1, 2, 3, 4, 5]:
                file = cifar10_dir / f'data_batch_{batch}'
                with file.open('rb') as f:
                    d = pickle.load(f, encoding='bytes')
                data = d[b'data']
                data = data.reshape(10000, 3, 32, 32)
                start = (batch - 1) * 10000
                end = batch * 10000
                self.data[start:end] = np.transpose(data, [0, 2, 3, 1])
                labels = d[b'labels']
                for i in range(10000):
                    self.meta.append(ImageNetMeta(f'c10_{batch}_{i}.png', None, labels[i], None))
        elif split == 'eval':
            file = cifar10_dir / f'test_batch'
            with file.open('rb') as f:
                d = pickle.load(f, encoding='bytes')
                data = d[b'data']
                data = data.reshape(10000, 3, 32, 32)
                self.data = np.transpose(data, [0, 2, 3, 1])
                labels = d[b'labels']
                for i in range(10000):
                    self.meta.append(ImageNetMeta(f'c10_test_{i}.png', None, labels[i], None))
        else:
            raise Exception

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.data[index], self.meta[index]
