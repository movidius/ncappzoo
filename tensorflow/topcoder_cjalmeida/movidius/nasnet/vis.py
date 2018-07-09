from pathlib import Path

import numpy as np
import tensorflow as tf

from movidius import cfg
from movidius.imagenet import ILSVRCDataset
from movidius.splits import dataset_from_split
from .transform import preprocess_nasnet_mobile_eval

tf.logging.set_verbosity(tf.logging.INFO)


def interactive():
    import matplotlib.pyplot as plt
    print('Loading dataset')
    eval_ds, num_classes = dataset_from_split('imagenet_val')

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                print('Load from meta graph')
                restorer = tf.train.import_meta_graph(str(cfg.NASNET_MOBILE_META))
                print("Restoring from disk")
                restorer.restore(sess, str(cfg.NASNET_MOBILE_CHECKPOINT))
                images = sess.graph.get_tensor_by_name('images:0')
                predictions = sess.graph.get_tensor_by_name('final_layer/predictions:0')
                print("Model restored")

                for i, (img, meta) in enumerate(eval_ds):
                    # import ipdb; ipdb.set_trace()
                    img = preprocess_nasnet_mobile_eval(img, meta)
                    batch = np.expand_dims(img, axis=0)
                    out = predictions.eval(feed_dict={images: batch})
                    truth_wnid = meta.wnid
                    cat = eval_ds.synsets[truth_wnid]
                    print('-----')
                    print(f'Image: {meta.key}, Class: {truth_wnid} - {cat}')
                    top5_cls = np.argsort(out.ravel())[::-1][:5]
                    top5_prob = np.sort(out.ravel())[::-1][:5]
                    for j in range(5):
                        cls = top5_cls[j]
                        p_wnid = eval_ds.label_to_wnid[cls]
                        p_cat = eval_ds.synsets[p_wnid]
                        print(f'Rank {j}: (p:{top5_prob[j]:.2f}) {p_wnid} - {p_cat}')

                    p = plt.imshow(eval_ds[i][0])
                    plt.show(block=True)
