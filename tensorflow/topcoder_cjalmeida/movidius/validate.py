from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Callable

import tensorflow as tf
from tqdm import tqdm

from movidius.dataset.utils import BatchDataset, Feeder, imagenet_coalesce_fn
from movidius.preprocessing import preprocess_eval, preprocess_train

from movidius.splits import dataset_from_split


class ValidateConfig:
    meta_file: Path = None
    images_name = 'images'
    logits_name = 'logits'
    proba_name = 'proba'
    eval_split = 'imagenet_val'
    checkpoint_file: Path
    img_shape: Tuple[int, int]
    build_fn: Callable[[tf.Tensor, int], Tuple[tf.Tensor, tf.Tensor]]
    loss_fn = None
    batch_size: int = None


def do_validate(cfg: ValidateConfig):
    print('Loading dataset')
    val_ds, num_classes = dataset_from_split(cfg.eval_split)
    batch_ds = BatchDataset(val_ds, batch_size=cfg.batch_size)
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            if cfg.meta_file:
                # saver = tf.train.import_meta_graph(str(cfg.meta_file))
                raise NotImplementedError
            elif cfg.build_fn:
                print('Building network')
                images = tf.placeholder(tf.float32, (None, cfg.img_shape[0], cfg.img_shape[1], 3), name='images')
                labels = tf.placeholder(tf.float32, (None, num_classes))
                logits, proba = cfg.build_fn(images=images, num_classes=num_classes, is_training=False)
                saver = tf.train.Saver(tf.global_variables())
            else:
                raise Exception('Missing meta_file or build_fn from config')

            feeder = Feeder(batch_ds, imagenet_coalesce_fn(num_classes, preprocess_eval, *cfg.img_shape))
            # accuracy metrics
            if cfg.loss_fn is None:
                cfg.loss_fn = tf.losses.softmax_cross_entropy
            loss = cfg.loss_fn(onehot_labels=labels, logits=logits, scope='loss')
            target_vec = tf.argmax(labels, axis=1)
            _, top1_upd = tf.metrics.mean(tf.cast(tf.nn.in_top_k(proba, target_vec, 1), tf.float32))
            _, top5_upd = tf.metrics.mean(tf.cast(tf.nn.in_top_k(proba, target_vec, 5), tf.float32))
            _, loss_upd = tf.metrics.mean(loss)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                print('Restoring from checkpoint')
                saver.restore(sess, str(cfg.checkpoint_file))

                bar = tqdm(total=len(val_ds))
                feeder.shuffle()
                it = iter(feeder)

                while True:
                    try:
                        _images, _labels = next(it)
                    except StopIteration:
                        break
                    top1_v, top5_v, loss_v = sess.run([top1_upd, top5_upd, loss_upd],
                                                      feed_dict={images: _images, labels: _labels})
                    bar.update(len(_labels))
                    pf = OrderedDict([
                        ('top1', f'{top1_v: .4f}'),
                        ('top5', f'{top5_v: .4f}'),
                        ('loss', f'{loss_v: .4f}')
                    ])
                    bar.set_postfix(pf)
                bar.close()
