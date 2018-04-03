from pathlib import Path
from typing import Tuple, Callable
from multiprocessing import Process
import tempfile

from movidius.cfg import NCS_NCPROFILE
from movidius.splits import dataset_from_split
import tensorflow as tf


class ProfileConfig:
    shape: Tuple[int, int] = None
    build_fn: Callable = None
    output_node: str = 'proba'
    split: str = 'eval'
    temp_dir: Path = None


def _meta_file(cfg: ProfileConfig) -> Path:
    return cfg.temp_dir / 'network.meta'


def do_profile(cfg: ProfileConfig):
    import atexit, shutil
    if cfg.temp_dir is None:
        cfg.temp_dir = Path(tempfile.mkdtemp())
        atexit.register(lambda *args, **kwargs: shutil.rmtree(str(cfg.temp_dir.absolute())))
    else:
        cfg.temp_dir.mkdir(exist_ok=True)

    # build meta file
    p = Process(target=_build_meta, args=(cfg,))
    p.start()
    p.join()

    # profile
    _run_profile(cfg)


def _run_profile(cfg: ProfileConfig):
    from posix_spawn import posix_spawn
    import os
    print('Running NCS profile tool')
    exec = bytes(NCS_NCPROFILE, 'utf-8')
    args = ['-s', '12',
            str(_meta_file(cfg)),
            '-in', 'images',
            '-on', cfg.output_node]
    args = [exec] + [bytes(arg, 'utf-8') for arg in args]
    print(args)
    pid = posix_spawn(exec, args)
    ret = os.waitpid(pid, 0)
    assert ret[1] == 0


def _build_meta(cfg: ProfileConfig):
    print('Building meta graph')
    _, num_classes = dataset_from_split(cfg.split)
    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder("float", [1, cfg.shape[0], cfg.shape[1], 3], name='images')
        cfg.build_fn(images=images, num_classes=num_classes, is_training=False)
        var_list = tf.global_variables()
        logdir = cfg.temp_dir / 'log'
        logdir.mkdir(exist_ok=True)
        writer = tf.summary.FileWriter(str(logdir))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
            saver = tf.train.Saver(var_list)
            saver.save(sess, str(_meta_file(cfg).with_suffix('')))
    del sess
