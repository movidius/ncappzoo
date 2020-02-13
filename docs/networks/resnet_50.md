---
layout: default
title: resnet_50
parent: Networks
---
# resnet_50
## Introduction
The [resnet_50](https://github.com/opencv/open_model_zoo/blob/master/models/public/resnet-50/resnet-50.md) network can be used for image classification.  

This sample utilizes the OpenVINO Inference Engine from the [OpenVINO Deep Learning Development Toolkit](https://software.intel.com/en-us/openvino-toolkit) and was tested with the 2020.1 release.


The provided Makefile does the following

1. Downloads the Caffe prototxt file and makes any changes necessary to work with the OpenVINO toolkit (tested with 2020.1) and the Intel Neural Compute Stick (NCS1/NCS2). 
2. Downloads and generates the required ilsvrc12 data.
3. Downloads the .caffemodel file from [Open Model Zoo](https://github.com/opencv/open_model_zoo).
4. Compiles the network into an IR (intermediate representation) format file (.xml) using the OpenVINO Model Optimizer. 
5. Runs a single inference on an image as an example of how to use this network with the OpenVINO Python API using the simple_classifier_py app.

## Model Information
### Inputs
 - name: 'data', shape: [1x3x224x224], Expected color order is BGR.
### Outputs 
 - name: 'prob', shape: [1, 1000] - Output indexes represent each class probability.


## Running this Example
~~~
make run
~~~

## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run or make run_py
Runs a sample application with the network.

### make cpp
Builds the C++ example program run_cpp which can be executed with make run_cpp. 

### make run_cpp
Runs the provided run_cpp executable program that is built via make cpp.  This program sends a single image to the Neural Compute Stick and receives and displays the inference results.

### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data, compile_model, compile_cpp.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.

