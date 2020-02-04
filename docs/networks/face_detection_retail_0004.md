---
layout: default
title: face_detection_retail_0004

parent: Networks
---
# face_detection_retail_0004
## Introduction
The [face_detection_retail_0004](https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-retail-0004/description/face-detection-retail-0004.md) network can be used for face detection. This model can be used to align faces for use with face recognition.

This sample utilizes the OpenVINO Inference Engine from the [OpenVINO Deep Learning Development Toolkit](https://software.intel.com/en-us/openvino-toolkit) and was tested with the 2020.1 release.

The provided Makefile does the following

1. Downloads the IR files from the [Open Model Zoo](https://github.com/opencv/open_model_zoo)
2. Takes an image and runs an inference on the face-detection-retail-0004 model.

The sample can also be used to crop images and write them to file. 

## Model Information
### Inputs
 - name: 'input', shape: [1x3x300x300], Expected color order is BGR.
### Outputs 
 - name: 'detection_out', shape: [1, 1, N, 7] - where N is the number of detected bounding boxes. For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max].

## Running this Example
~~~
make run
~~~

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

