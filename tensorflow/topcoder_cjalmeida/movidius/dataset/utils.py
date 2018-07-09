import hashlib
import json
from queue import Empty
from multiprocessing import Queue, Process, cpu_count
from typing import List

import numpy as np
from ..preprocessing import batch, one_hot


def md5sum(obj):
    hash = bytes(json.dumps(obj), 'utf-8')
    return hashlib.md5(hash).hexdigest()


class MergeDataset:
    def __init__(self, sources):
        self.sources = sources
        self.indexes = []
        for src_idx in range(len(sources)):
            for item_idx in range(len(sources[src_idx])):
                self.indexes.append((src_idx, item_idx))

    def shuffle(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        src_idx, item_idx = self.indexes[index]
        return self.sources[src_idx][item_idx]


class BatchDataset:
    def __init__(self, src, batch_size):
        self.src = src
        self.batch_size = batch_size
        self.indexes = np.arange(0, len(self.src)).tolist()
        self.batches = list(batch(self.indexes, self.batch_size))

    def shuffle(self):
        np.random.shuffle(self.indexes)
        self.batches = list(batch(self.indexes, self.batch_size))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch_idx = self.batches[index]
        data = [self.src[idx] for idx in batch_idx]
        return data


def imagenet_coalesce_fn(num_classes, preprocess_fn, img_h, img_w, return_meta=False):
    def coalesce(data):
        data = [(preprocess_fn(img, meta, img_h, img_w), meta.label, meta) for img, meta in data]
        _images, _labels, _metas = zip(*data)
        _images = np.stack(_images)
        _labels = one_hot(_labels, num_classes)
        if return_meta:
            return _images, _labels, _metas
        else:
            return _images, _labels

    return coalesce


_EXIT = '##EXIT##'


def feeder_worker(dataset, coalesce_fn, work_queue: Queue, res_queue: Queue):
    while True:
        work_item = work_queue.get()
        if work_item == _EXIT:
            break

        else:
            data = dataset[work_item]
            res_queue.put(coalesce_fn(data))


class Feeder:
    def __init__(self, dataset, coalesce_fn, workers_count=cpu_count() - 1, infinite=False,
                 infinite_shuffle=True) -> None:
        self.dataset = dataset
        self.coalesce_fn = coalesce_fn
        self.res_queue = Queue(maxsize=10)
        self.work_queue = Queue()
        self.workers: List[Process] = []
        self.workers_count = workers_count
        self.shuffle_next = False
        self.remaining: int = None
        self.infinite = infinite
        self.infinite_shuffle = infinite_shuffle

    def __iter__(self):
        self.remaining = len(self.dataset)
        self.retrieved = 0

        # put jobs
        indexes = np.arange(0, len(self.dataset))
        if self.shuffle_next:
            shuffle_func = getattr(self.dataset, 'shuffle', None)
            if shuffle_func:
                shuffle_func()
            np.random.shuffle(indexes)
        for idx in indexes:
            self.work_queue.put(idx)

        for _ in range(self.workers_count):
            worker = Process(target=feeder_worker, name='feed_worker',
                             args=(self.dataset, self.coalesce_fn, self.work_queue, self.res_queue))
            self.workers.append(worker)
            worker.start()

        return self

    def __next__(self):
        if self.retrieved >= self.remaining:
            self.stop()
            if self.infinite:
                if self.infinite_shuffle:
                    self.shuffle_next = True
                iter(self)
                return next(self)
            else:
                raise StopIteration

        item = self.res_queue.get()
        self.retrieved += 1
        return item

    def shuffle(self):
        self.shuffle_next = True

    def stop(self):
        for _ in self.workers:
            self.work_queue.put(_EXIT)

        for worker in self.workers:
            worker.join()

        while not self.work_queue.empty():
            try:
                self.work_queue.get(timeout=0.5)
            except Empty:
                pass
        self.workers.clear()

    def __del__(self):
        self.stop()
