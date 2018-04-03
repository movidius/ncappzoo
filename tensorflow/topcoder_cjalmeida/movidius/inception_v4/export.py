import tensorflow as tf
import tensorflow.contrib.slim as slim
from .inception_v4 import inception_v4_arg_scope, inception_v4


def export():
    checkpoint_file = 'checkpoints/inception_v4/inception_v4.ckpt'
    meta_file = 'checkpoints/inception_v4/inception_v4.meta'
    log_dir = 'checkpoints/inception_v4/log'
    arg_scope_fn = inception_v4_arg_scope
    build_fn = inception_v4

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            images = tf.placeholder(tf.float32, (1, 224, 224, 3), name='images')

            print('Building network')
            with slim.arg_scope(arg_scope_fn()):
                logits, _ = build_fn(images, 1001, is_training=False, dropout_keep_prob=1.0, create_aux_logits=False)

            print('Build saver')
            restorer = tf.train.Saver()

            with tf.Session() as sess:
                print("Restoring from disk")
                restorer.restore(sess, str(checkpoint_file))
                print("Model restored")

                print("Export meta")
                restorer.export_meta_graph(str(meta_file), clear_devices=True)

                print("Write graph to tensorboard logdir")
                writer = tf.summary.FileWriter(log_dir)
                writer.add_graph(sess.graph)

