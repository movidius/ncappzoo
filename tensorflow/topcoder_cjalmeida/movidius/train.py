import shutil
from collections import OrderedDict, deque
from pathlib import Path
from typing import Callable, Tuple, List, Dict, Union

import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from movidius.dataset.utils import BatchDataset, imagenet_coalesce_fn, Feeder
from movidius.preprocessing import preprocess_train, preprocess_eval
from movidius.splits import dataset_from_split
import tensorflow.contrib.slim as slim


class TrainConfig:
    build_fn: Callable[[tf.Tensor], Tuple[tf.Tensor, Dict]] = None
    pretrained_var_list_fn: Callable = None
    pretrained_ckpt: str = None
    loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = None
    img_shape: Tuple[int, int] = None
    batch_size: int = None
    fine_tune_var_list_fn: Callable = None
    learning_rate: Union[float, tf.Tensor] = 0.001
    momentum = 0.9
    weight_decay = 0
    optimizer: str = 'adam'
    train_split: str = 'train'
    eval_split: str = 'eval'
    work_dir: Path = None
    epochs: int = None
    reset_optim: bool = False
    preprocess_train_fn: Callable = None
    preprocess_eval_fn: Callable = None
    adjust_optimizer_fn: Callable = None
    eval_losses_max_samples: int = None
    save_best = True
    eval_freq = 30
    summary_freq = 50
    force_is_training: bool = None
    custom_initializer = None


def _restore_from_checkpoint(sess, last_checkpoint, var_lists):
    """ try to restore from checkpoint in var_lists order of preference """
    last = var_lists[-1]
    for var_list in var_lists:
        try:
            restorer = tf.train.Saver(var_list, name='restorer')
            restorer.restore(sess, last_checkpoint)
            break
        except:
            if var_list == last:
                raise


def _log_scalar(writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def noop_adjust_optimizer(**kwargs):
    pass


def do_train(cfg: TrainConfig):
    cfg.work_dir.mkdir(exist_ok=True, parents=True)

    train_ds, num_classes = dataset_from_split(cfg.train_split)
    eval_ds, _ = dataset_from_split(cfg.eval_split)

    with tf.device('/gpu:0'):
        graph = tf.Graph()
        with graph.as_default():
            # working variables
            images = tf.placeholder(tf.float32, (None, cfg.img_shape[0], cfg.img_shape[1], 3), name='images')
            labels = tf.placeholder(tf.float32, (None, num_classes), name='labels')
            is_training = tf.placeholder(tf.bool, (), name='is_training')
            global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
            global_epoch = tf.Variable(0, dtype=tf.int64, name='global_epoch', trainable=False)
            global_step_inc_op = tf.assign_add(global_step, 1)

            print('Building network')
            logits, proba = cfg.build_fn(images=images, num_classes=num_classes, is_training=is_training)

            pretrained_var_list = cfg.pretrained_var_list_fn() if cfg.pretrained_var_list_fn else None

            logdir = cfg.work_dir / 'log'
            logdir.mkdir(exist_ok=True)
            text_log = (logdir / 'log.txt').open('a')

            def log_text(text):
                prefix = datetime.datetime.utcnow().isoformat()
                text_log.write(f'{prefix}: {text}\n')
                tqdm.write(text)

            print('Logging to tensorboard at %s' % logdir)

            writer = tf.summary.FileWriter(str(logdir))
            writer.add_graph(graph)

            # fine-tune optimizer
            var_list = None
            if cfg.fine_tune_var_list_fn is not None:
                var_list = cfg.fine_tune_var_list_fn()

            save_var_list_without_optim = tf.global_variables()

            # loss func
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if cfg.loss_fn is None:
                    cfg.loss_fn = tf.losses.softmax_cross_entropy
                loss = cfg.loss_fn(onehot_labels=labels, logits=logits, scope='loss')

                if cfg.optimizer == 'adam':
                    log_text('Using Adam optimizer')
                    optim = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate)
                elif cfg.optimizer == 'sgd':
                    log_text('Using SGD with momentum and Nesterov optimizer')
                    optim = tf.train.MomentumOptimizer(learning_rate=cfg.learning_rate, momentum=cfg.momentum,
                                                       use_nesterov=False)
                elif cfg.optimizer == 'yellowfin':
                    log_text('Using Yellowfin optimizer')
                    from .yellowfin import YFOptimizer
                    optim = YFOptimizer(learning_rate=cfg.learning_rate, momentum=cfg.momentum)
                else:
                    raise Exception

                # train_op = optim.minimize(loss, global_step=global_step, var_list=var_list)
                train_op = slim.learning.create_train_op(loss, optim, variables_to_train=var_list,
                                                         global_step=global_step)

            # all relevant to train variables are in context
            save_var_list = tf.global_variables()

            # accuracy metrics
            proba_eval = proba
            target_vec = tf.argmax(labels, axis=1)
            top1_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(proba_eval, target_vec, 1), tf.float32))
            top5_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(proba_eval, target_vec, 5), tf.float32))

            # add summaries
            top1_sum = tf.summary.scalar('top1', top1_op)
            top5_sum = tf.summary.scalar('top5', top5_op)
            images_sum = tf.summary.image('sample_input', images)
            merge_sum = tf.summary.merge([top1_sum, top5_sum, images_sum])
            loss_sum = tf.summary.scalar('loss', loss)

            print('Loading datasets')
            # batch train
            batch_ds = BatchDataset(train_ds, cfg.batch_size)
            if cfg.preprocess_train_fn is None:
                cfg.preprocess_train_fn = preprocess_train
            feeder = Feeder(batch_ds, imagenet_coalesce_fn(num_classes, cfg.preprocess_train_fn, *cfg.img_shape))

            # eval ds
            eval_batch_ds = BatchDataset(eval_ds, cfg.batch_size)
            if cfg.preprocess_eval_fn is None:
                cfg.preprocess_eval_fn = preprocess_eval
            eval_feeder = Feeder(eval_batch_ds,
                                 imagenet_coalesce_fn(num_classes, cfg.preprocess_eval_fn, *cfg.img_shape),
                                 infinite=True, infinite_shuffle=True)
            eval_feeder_it = iter(eval_feeder)

            if cfg.adjust_optimizer_fn is None:
                cfg.adjust_optimizer_fn = noop_adjust_optimizer

            saver = tf.train.Saver(save_var_list, name='saver')
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if cfg.custom_initializer is not None:
                    sess.run(cfg.custom_initializer)
                savedir = cfg.work_dir / 'save'
                savefile = savedir / 'model'
                last_checkpoint = tf.train.latest_checkpoint(savedir)
                if last_checkpoint:
                    log_text('Loading from existing checkpoint at %s' % savefile)
                    _var_list = save_var_list_without_optim if cfg.reset_optim else save_var_list
                    _restore_from_checkpoint(sess, last_checkpoint, [_var_list])

                elif cfg.pretrained_ckpt is not None:
                    log_text('Starting training from a pre-trained checkpoint at %s' % cfg.pretrained_ckpt)
                    assert pretrained_var_list is not None
                    savedir.mkdir(exist_ok=True)
                    restorer = tf.train.Saver(pretrained_var_list, name='restorer')
                    restorer.restore(sess, str(cfg.pretrained_ckpt))
                    sess.run(tf.assign(global_step, 0))
                else:
                    log_text('Starting training from scratch')

                eval_max_samples = cfg.eval_losses_max_samples
                if eval_max_samples is None:
                    eval_max_samples = len(eval_batch_ds)
                losses_eval = deque(maxlen=eval_max_samples)
                top1_eval = deque(maxlen=eval_max_samples)
                top5_eval = deque(maxlen=eval_max_samples)
                best_eval = None

                is_training_train = True if cfg.force_is_training is None else cfg.force_is_training
                is_training_eval = False if cfg.force_is_training is None else cfg.force_is_training

                start_epoch = global_epoch.eval(sess)
                for epoch in range(start_epoch, cfg.epochs):
                    log_text(f'Training epoch {epoch+1}/{cfg.epochs}')
                    losses = []
                    bar = tqdm(total=len(train_ds))
                    feeder.shuffle()
                    for _images, _labels in feeder:
                        _, loss_v, loss_sum_v = sess.run([train_op, loss, loss_sum],
                                                               feed_dict={images: _images, labels: _labels,
                                                                          is_training: is_training_train})
                        step = global_step.eval(sess)
                        losses.append(loss_v)
                        bar.update(len(_labels))
                        writer.add_summary(loss_sum_v, step)

                        # lower frequency summaries
                        if step % cfg.summary_freq == 0:
                            summary = sess.run(merge_sum,
                                               feed_dict={images: _images, labels: _labels,
                                                          is_training: is_training_eval})
                            writer.add_summary(summary, step)

                        if step > 0 and step % cfg.eval_freq == 0:
                            _images, _labels = next(eval_feeder_it)
                            top1_v, top5_v, loss_v = sess.run([top1_op, top5_op, loss],
                                                              feed_dict={images: _images, labels: _labels,
                                                                         is_training: is_training_eval})
                            losses_eval.append(loss_v)
                            top1_eval.append(top1_v)
                            top5_eval.append(top5_v)

                            _log_scalar(writer, 'top1_val', np.mean(top1_eval), step)
                            _log_scalar(writer, 'top5_val', np.mean(top5_eval), step)
                            _log_scalar(writer, 'losses_val', np.mean(losses_eval), step)

                            cfg.adjust_optimizer_fn(**locals())
                        bar.set_postfix(
                            OrderedDict([
                                ('loss', f'{np.mean(losses):.5f}'),
                                ('step', str(step)),
                                ('val_loss', f'{np.mean(losses_eval):.5f}' if losses_eval else '0'),
                                ('val_top1', f'{np.mean(top1_eval):.5f}' if top1_eval else '0'),
                                ('val_top5', f'{np.mean(top5_eval):.5f}' if top5_eval else '0')
                            ]))

                        step += 1
                    bar.close()
                    sess.run(tf.assign_add(global_epoch, 1))
                    if not cfg.save_best or best_eval is None or np.mean(losses_eval) < best_eval:
                        log_text(f'Saving model to at step {step}, epoch {epoch} %s' % savefile)
                        saver.save(sess, str(savefile), global_step=global_step)
                        best_eval = np.mean(losses_eval)

                    log_text(f'Validation metrics for epoch {epoch}: ' + str([
                        f'val_loss: {np.mean(losses_eval):.5f}',
                        f'val_top1: {np.mean(top1_eval):.5f}',
                        f'val_top5: {np.mean(top5_eval):.5f}'
                    ]))
                    log_text('----')
