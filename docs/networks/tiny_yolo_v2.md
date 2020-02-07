---
layout: default
title: tiny_yolo_v2
parent: Networks
---
# tiny_yolo_v2
## Introduction
The TinyYolo V2 network can be used for object recognition and classification.  See [https://pjreddie.com/darknet/yolov2/](https://pjreddie.com/darknet/yolov2/) for more information on this network. This model was trained with the Pascal VOC dataset and can detect 20 different objects.

The provided Makefile does the following
1. Clones the Darkflow repo.
2. Downloads the Tiny Yolo v2 cfg and weights files.
3. Converts the cfg and weights files to a Tensorflow pb file.
4. Converts the Tensorflow pb file to an OpenVINO IR file.
5. Loads the IR file and runs the provided tiny_yolo_v2.py program that does a single inference on a provided image as an example on how to use the network using the OpenVINO 2020.1 Inference Engine


## Prerequisites
1. NCS2 device
2. OpenVINO 2020.1
3. Cython (can be installed via the command 'make install-reqs')
4. Darkflow (can be installed via the command 'make install-reqs')

## Model Information
### Inputs
 - name: 'input', shape: [1x3x416x416], Expected color order is BGR. Original network expects RGB, but for this sample, the IR is compiled with the --reverse_input_channels option to convert the IR to expect the BGR color order.
### Outputs 
 - name: 'output/YoloRegion', shape: [1, 21125]. 
 
 **Note**:The inference results come back in a 21125 element array which the app explicitly reshapes into a [5, 25, 13, 13] shaped array. Afterwards, the app transposes, using (2,3,0,1) order, the array into [13, 13, 5, 25] order then does another reshape into a [169, 5, 25] shape for a more intuitive way of processing of values.  Tiny Yolo v2 uses a 13 x 13 grid with 5 anchor boxes per grid cell. For each anchor box, there are 25 values (number of classes (20) + the bounding box coordinates (4) + the object score (1)). 


For the post processing, we are checking each grid cell, then in that grid cell, we are checking each anchor box. And in each anchor box, we are checking the object score (1) and class scores (20). 


All scores will need to be filtered to achieve desired results. Please see Algorithm Thresholds section. 

## Running this Example
~~~
make run
~~~
**Note**: You can specify images using the INPUT variable. Example: ```make run INPUT=../../data/images/cat.jpg```
**Note**: If you are not seeing the results you desire, you can adjust the IOU_THRESHOLD and default detection threshold.


## Algorithm Thresholds
There are a few thresholds in the code you may want to tweek if you aren't getting results that you expect:
- <strong>threshold</strong>: This is the minimum probability allowed for boxes returned from tiny yolo v2.  This should be between 0.0 and 1.0.  A lower value will allow more boxes to be displayed.
- <strong>IOU_THRESHOLD</strong>: Determines which boxes from Tiny Yolo v2 should be separate objects vs identifying the same object.  This is based on the intersection-over-union calculation.  The closer this is to 1.0 the more similar the boxes need to be in order to be considered around the same object.


## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run or make run_py
Runs a sample application with the network.

### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data, compile_model.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed. 

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.
