---
layout: default
title: ssd_mobilenet_v1_caffe
parent: Networks
---
# ssd mobilenet_v1_caffe
## Introduction
The [ssd mobilenet v1 caffe](https://github.com/chuanqi305/MobileNet-SSD) network can be used for object detection and can detect 20 different types of objects (This model was pre-trained with the Pascal VOC dataset). More information can be found at: https://github.com/chuanqi305/MobileNet-SSD. The list of objects that this network can detect are:

```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

This sample utilizes the OpenVINO Inference Engine from the [OpenVINO Deep Learning Development Toolkit](https://software.intel.com/en-us/openvino-toolkit) and was tested with the 2020.1 release.

The provided Makefile does the following

1. Downloads the prototxt and caffe weight files using the model downloader from the [Open Model Zoo.](https://github.com/opencv/open_model_zoo)
2. Compiles an IR (Intermediate Representation) for the model.
3. Takes an image/camera input, loads the IR file, and runs an inference using the SSD Mobilenet model.

## Model Information
### Inputs
 - name: 'data', shape: [1x3x300x300], Expected color order is BGR.
### Outputs 
 - name: 'detection_out', shape: [1, 1, N, 7] - where N is the number of detected bounding boxes. For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max].


## Running this Example
~~~
make run
~~~

**Note**: This sample can also be used with a web cam. To run with a webcam, use the following command: ```make run INPUT=cam```
 
## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run or make run_py
Runs a sample application with the network.

### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.

