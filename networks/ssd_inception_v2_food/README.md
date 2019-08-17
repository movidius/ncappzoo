# SSD Inception V2 Grocery Detection

## Introduction
The SSD Inception V2 network can be used to detect a number of objects specified by a particular training set. This model in particular can detect the following produce/items usually found in grocery stores:
- bananas
- bell peppers
- tomatoes
- broccoli
- cheese
- steak
- beef
- corn
- eggs
- olive oil

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
The example runs inference using a video stream taken from an existing camera device. Bounding boxes and their associated probabilities and classifications are rendered on frames of the video stream and presented on a OpenCV frame.


## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run
Runs a sample application with the FP16 network. Users must plug in their Intel Neural Compute Stick 2 in order to successfully run this application.

### make run_py
Runs the `ssd_inception_v2_food.py` script which takes in a video stream from an existing camera, runs inference on each frame, and renders the output to the user.

### make run_FP32
Runs the `ssd_inception_v2_food.py` script with the FP32 network. Note that this application will only run successfully on CPUs that OpenVINO supports.

### make run_FP16
Runs the `ssd_inception_v2_food.py` script with the FP16 network. Users must plug in their Intel Neural Compute Stick 2 in order to successfully run this application.

### make train
**TO BE IMPLEMENTED.** Trains a SSD Inception V2 model using the Tensorflow Object Detection API given an `Annotations` and `JPEGImages` folder containing .xml and .jpg images, respectively, for training. Training is not necessary since the sample will download a pre-trained model. This option allows for the user to further refine the SSD Inception V2 model if they so desire.

### make help
Shows makefile possible targets and brief descriptions.

### make all
Makes the follow items: deps, data, compile_model.

### make compile_model
Compiles the trained model to generate a OpenVINO IR file.  This file can be loaded on the Neural Compute Stick for inferencing.

### make get_model
Downloads the trained model.

### make deps
Downloads and prepares a trained network for compilation with the OpenVINO toolkit.

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.
