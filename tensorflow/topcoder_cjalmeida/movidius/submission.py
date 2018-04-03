from collections import OrderedDict
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
import pystache
import pickle
from tqdm import tqdm

from movidius.cfg import NCS_NCCOMPILE
from subprocess import check_call
import tensorflow as tf

from movidius.dataset.imagenet import ImageNetMeta
from movidius.dataset.utils import BatchDataset, Feeder, imagenet_coalesce_fn
from movidius.splits import dataset_from_split
from .inferences import MovidiusImage, write_inferences_csv, score_inferences


class SubmissionConfig:
    model_name: str = None
    shape: Tuple[int, int] = None
    build_fn: Callable = None
    preprocess_fn: Callable = None
    weights_file: Path = None
    submission_dir: Path = None
    output_node: str = 'proba'
    test_split: str = 'test'
    num: int = 0
    skip_compile: bool = False
    skip_inference: bool = False
    skip_upload: bool = False
    score: bool = False
    limit: int = None
    compile_tool: str = NCS_NCCOMPILE
    custom_test_dataset_fn: Callable = None
    custom_inferences_file: Path = None


class ScoreConfig:
    split: str = None
    shape: Tuple = None
    build_fn: Callable = None
    preprocess_fn: Callable = None
    weights_file: Path = None
    time: float = None
    batch_size: int = 32
    limit: int = None


def do_score_gpu(cfg: ScoreConfig):
    print('Building meta graph')
    ds, num_classes = dataset_from_split(cfg.split)
    assert cfg.time, "Please provide estimate reference time."
    with tf.device('/gpu:0'):
        g = tf.Graph()
        with g.as_default():
            predictions = []
            images = tf.placeholder("float", [None, cfg.shape[0], cfg.shape[1], 3], name='images')
            logits, proba = cfg.build_fn(is_training=False, images=images, num_classes=num_classes)
            var_list = tf.global_variables()
            with tf.Session() as sess:
                restorer = tf.train.Saver(var_list, reshape=False)
                restorer.restore(sess, str(cfg.weights_file))
                batch_ds = BatchDataset(ds, cfg.batch_size)
                feeder = Feeder(batch_ds,
                                imagenet_coalesce_fn(num_classes, cfg.preprocess_fn, *cfg.shape, return_meta=True))
                bar = tqdm(total=len(ds))
                metrics = Metrics()
                count = 0
                for _images, _labels, _metas in feeder:
                    proba_v = sess.run(proba, feed_dict={images: _images})
                    for i in range(len(_labels)):
                        image = MovidiusImage(_metas[i].key, _metas[i].file, _metas[i].label)
                        image.save_top_k(proba_v[i], 5)
                        image.inference_time = cfg.time
                        predictions.append(image)
                        bar.update()
                        if _metas[i].label is not None:
                            metrics.add(image)
                            bar.set_postfix(metrics.as_postfix())

                    count += len(_labels)
                    if cfg.limit and count >= cfg.limit:
                        break
                print('Score: %.2f' % score_inferences(predictions))


def do_submit(cfg: SubmissionConfig):
    from multiprocessing import Process

    if (not cfg.score) and (not cfg.num):
        raise Exception('--num is required when uploading')

    if not cfg.skip_compile:
        # build the meta graph file
        # note, running in a subprocess to make sure tensorflow releases gpu memory after call
        p = Process(target=build_meta, args=(cfg,))
        p.start()
        p.join()

        # compile to movidius graph
        nc_compile(cfg)

    # do inference
    if not cfg.skip_inference:
        run_inference(cfg)

    # pack submission archive
    if not cfg.score:
        pack(cfg)

        # upload to S3
        if not cfg.skip_upload:
            upload(cfg)

    else:
        score(cfg)


def build_meta(cfg: SubmissionConfig):
    print('Building meta graph')
    _, num_classes = dataset_from_split(cfg.test_split)
    cfg.submission_dir.mkdir(exist_ok=True, parents=True)
    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder("float", [1, cfg.shape[0], cfg.shape[1], 3], name='images')
        cfg.build_fn(is_training=False, images=images, num_classes=num_classes)
        var_list = tf.global_variables()
        with tf.Session() as sess:
            restorer = tf.train.Saver(var_list, reshape=False)
            restorer.restore(sess, str(cfg.weights_file))
            saver = tf.train.Saver(var_list)
            saver.save(sess, str(_meta_file(cfg).with_suffix('')))
    del sess


def nc_compile(cfg: SubmissionConfig):
    from posix_spawn import posix_spawn
    import os
    print('Compiling to NCS blob')
    cfg.submission_dir.mkdir(exist_ok=True, parents=True)
    exec = bytes(cfg.compile_tool, 'utf-8')
    args = ['-s', '12',
            str(_meta_file(cfg)),
            '-in', 'images',
            '-on', cfg.output_node,
            '-o', str(_graph_file(cfg))]
    args = [exec] + [bytes(arg, 'utf-8') for arg in args]
    env = os.environ.copy()
    env['PYTHONPATH'] = 'ncsdk/api/python'
    env = {bytes(k, 'utf-8'): bytes(v, 'utf-8') for k, v in env.items()}
    # pid = posix_spawn(exec, args, env=env)
    # ret = os.waitpid(pid, 0)
    ret = check_call(args, env=env)
    assert ret == 0


def _meta_file(cfg) -> Path:
    return cfg.submission_dir / 'network.meta'


def _graph_file(cfg: SubmissionConfig) -> Path:
    return cfg.submission_dir / 'compiled.graph'


def _inferences_file(cfg: SubmissionConfig) -> Path:
    return cfg.submission_dir / 'inferences.csv'


def _zip_file(cfg: SubmissionConfig) -> Path:
    return cfg.submission_dir / f'submission-{cfg.num}.zip'


def _images_file(cfg) -> Path:
    return cfg.submission_dir / f'images.pkl'


def run_inference(cfg: SubmissionConfig):
    from mvnc import mvncapi as mvnc
    print('Enumerating devices')
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        raise Exception('No devices found')
    print('Found %s devices' % len(devices))

    # pick first available device
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    print('Loading graph')
    graphfile = _graph_file(cfg).read_bytes()
    graph = device.AllocateGraph(graphfile)

    if cfg.custom_test_dataset_fn:
        print('Using custom dataset')
        ds, num_classes = cfg.custom_test_dataset_fn()
    else:
        print('Loading dataset split %s' % cfg.test_split)
        ds, num_classes = dataset_from_split(cfg.test_split)

    print('Starting inference on NCS...')
    images = []
    bar = tqdm(total=len(ds))
    metrics = Metrics()
    count = 0
    target_h, target_w = cfg.shape
    for img_data, meta in ds:  # type: np.ndarray, ImageNetMeta
        img_data = cfg.preprocess_fn(img_data, meta, target_h, target_w)
        assert graph.LoadTensor(img_data.astype(np.float16), 'user object')
        output, userobj = graph.GetResult()
        image = MovidiusImage(meta.key, meta.file, meta.label)
        image.save_top_k(output, 5)
        image.inference_time = np.sum(graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN))
        images.append(image)
        bar.update()
        if meta.label is not None:
            metrics.add(image)
            bar.set_postfix(metrics.as_postfix())

        count += 1
        if cfg.limit and count >= cfg.limit:
            break

    graph.DeallocateGraph()
    device.CloseDevice()

    if cfg.custom_inferences_file:
        infer_file = cfg.custom_inferences_file
    else:
        infer_file = _inferences_file(cfg)

    write_inferences_csv(str(infer_file), images)
    _images_file(cfg).write_bytes(pickle.dumps(images))


class Metrics:
    """ Helper to calculate running statistics on validation"""

    def __init__(self):
        self.total = 0
        self.match_k1 = 0
        self.match_k5 = 0
        self.time = 0

    def add(self, image: MovidiusImage):
        self.total += 1
        self.match_k1 += 1 if image.class_index == image.top_k[0][0] else 0
        self.match_k5 += 1 if image.class_index in [k[0] for k in image.top_k] else 0
        self.time += image.inference_time

    def as_postfix(self):
        return OrderedDict([
            ('top1', f'{(self.match_k1 / self.total): .4f}'),
            ('top5', f'{(self.match_k5 / self.total): .4f}'),
            ('time', f'{(self.time / self.total): .4f}'),
        ])


def pack(cfg: SubmissionConfig):
    from zipfile import ZipFile, ZIP_DEFLATED
    zip = ZipFile(_zip_file(cfg), mode='w', compression=ZIP_DEFLATED)

    # write submission outputs
    f = _inferences_file(cfg)
    zip.write(str(f), f.name)

    f = _graph_file(cfg)
    zip.write(str(f), f.name)

    f = _meta_file(cfg)
    zip.write(str(f), f.name)

    # write support files in python source
    src = [Path('environment.yml')]
    src += list(Path('movidius').rglob('*.py'))
    src += list(Path('.').glob('*.py'))
    for f in src:
        zip.write(str(f), 'supporting/' + str(f))

    # write README
    readme = Path('README').read_text()
    readme_v = pystache.render(readme, {'model_name': cfg.model_name, 'num': cfg.num})
    zip.writestr('supporting/README', readme_v)

    zip.close()


def upload(cfg):
    import boto3
    s3 = boto3.resource('s3')
    f = _zip_file(cfg)
    print(f'Uploading file to S3: https://s3.amazonaws.com/cj-movidius/{f.name}')
    s3.Object('cj-movidius', f.name).put(Body=f.read_bytes())


def score(cfg: SubmissionConfig):
    print('Scoring')
    images = pickle.loads(_images_file(cfg).read_bytes())
    print('Score: %.2f' % score_inferences(images))
