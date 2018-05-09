# tf_src

This directory is not a specific model, but rather a place where the tensorflow source will be downloaded if needed by the TensorFlow projects in the ncappzoo.

Many of the TensorFlow models in the ncappzoo require access to the TensorFlow git repository here: [https://github.com/tensorflow/tensorflow.git](https://github.com/tensorflow/tensorflow.git).  Running 'make all' in this directory will download that repository here.

If you already have this repository avaiable on your system you can set the TF_SRC_PATH environment variable to it's location on your system in which case this directory will not be populated.
