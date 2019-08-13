# simple_classifier_cpp
## Introduction
This application runs an inference on an image using [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).  Although the sample uses GoogLeNet as the default network, other classifier models can also be used. 

The provided Makefile does the following

1. Downloads the Caffe prototxt file and makes few changes necessary to work with the Intel® Neural Compute Stick (NCS1/NCS2) and the OpenVINO toolkit (tested with version 2019 R2).
2. Downloads and generates the required ilsvrc12 data.
3. Downloads the caffemodel weights file from the [Open Model Zoo](https://github.com/opencv/open_model_zoo).
3. Compiles the model to an IR (Intermediate Representation) format file using the Model Optimizer. An IR is a static representation of the model that is compatitible with the OpenVINO Inference Engine API. 
4. There is a run.py provided that does a single inference on a provided image as an example on how to use the network using the OpenVINO Inference Engine C++ API.

## Building the Example

To run the example code do the following :
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make all```

## Running the Example
To run the example code do the following :
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make run``` 


## Prerequisites
This program requires:
- 1 NCS device
- OpenVINO 2019 R2 Toolkit
- OpenCV 3.3 with Video for Linux (V4L) support and associated Python bindings*.

*It may run with older versions but you may see some glitches such as the GUI Window not closing when you click the X in the title bar, and other key binding issues.

Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make run or make run_cpp
Runs the sample application.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application.

### make data
Gathers all of the required data need to run the sample.

### make deps
Builds all of the dependencies needed to run the sample.

### make default_model
Compiles an IR file from a default model to be used when running the sample.

### make install_reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 
 
### make clean
Removes all the temporary files that are created by the Makefile.
