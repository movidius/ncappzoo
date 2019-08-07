# SSD Inception V2

## Introduction
The SSD Inception V2 network can be used to detect a number of objects specified by a particular training set. This model in particular can detection the following gestures:
- point up
- point down
- point left
- point right
- exposed palm
- closed fist with exposed bottom palm
- fist with exposed knuckles. 

The provided Makefile does the following:
1. Downloads a trained model
2. Downloads test images
3. Compiles the network using the OpenVINO Model Optimizer
4. There is a python example (run.py) which runs an inference for all of the test images to show how to use the network with the OpenVINO toolkit.

This network is based on the [TensorFlow Object Detection API  SSD Inception V2 model.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) The model was modified, trained, and saved in order to be compatible with the OpenVINO toolkit.


## Running this example

```
make run
```
The example runs an inference with the image `gesture.jpg`. Other gesture images can be found in the `training/JPEGImages` folder.


## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run
Runs a sample application with the network.

### make run_py
Runs the simple_classifier_py python script which sends a single image to the Neural Compute Stick and receives and displays the inference results.

### make train
**TO BE IMPLEMENTED.** Trains a SSD Inception V2 model for use with the sample. Training is not necessary since the sample will download a pre-trained model. This option allows for the user to further refine the SSD Inception V2 model if they so desire.

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
