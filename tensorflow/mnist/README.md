# Mnist
## Introduction
The mnist network can be used for handwriting recognition for the digits 0-9.  The provided Makefile does the following

1. Downloads a trained model 
2. Downloads test images
3. Compiles the network using the OpenVINO Model Optimizer.
4. There is a python example (run.py) which runs an inference for all of the test images to show how to use the network with the OpenVINO toolkit. 

This network is based on the [TensorFlow 1.4 mnist_deep.py example.](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py). The model was modified, trained, and saved in order to be compatible with the OpenVINO toolkit.

## Running this example
~~~
make run
~~~
The example runs an inference with the image one.png. Other digit images can be found in the data/digit_images folder.

## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run
Runs a sample application with the network.

### make run_py
Runs the simple_classifier_py python script which sends a single image to the Neural Compute Stick and receives and displays the inference results.

### make train
Trains a Mnist model for use with the sample. Training is not necessary since the sample will download a pre-trained model. This option allows for the user to further refine the Mnist model if they so desire. 


### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data, compile_model.

### make compile_model
Compiles the trained model to generate a OpenVINO IR file.  This file can be loaded on the Neural Compute Stick for inferencing. 

### make model
Downloads the trained model.

### deps 
Downloads and prepares a trained network for compilation with the OpenVINO toolkit.

### make clean
Removes all the temporary and target files that are created by the Makefile.

