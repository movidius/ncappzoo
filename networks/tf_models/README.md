# tf_models

This directory is not a specific model, but rather a place where the tensorflow models source will be downloaded if needed by the TensorFlow projects in the ncappzoo.

Many of the TensorFlow models in the ncappzoo require access to the TensorFlow models git repository here: [ https://github.com/tensorflow/models.git]( https://github.com/tensorflow/models.git).  Running 'make all' in this directory will download that repository here.

If you already have this repository avaiable on your system you can set the TF_MODELS_PATH environment variable to it's location on your system in which case this directory will not be populated.
