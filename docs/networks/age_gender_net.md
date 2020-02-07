---
layout: default
title: age_gender_net

parent: Networks
---
# age_gender_net
## Introduction
The [Age/GenderNet](https://github.com/opencv/open_model_zoo/blob/master/intel_models/age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md) network can be used for image classification. This model was trained to classify ages from 18-75. The model has 2 outputs, one for age: 'age-conv3' and another output for gender: 'prob'. 

This sample utilizes the OpenVINO Inference Engine from the [OpenVINO Deep Learning Development Toolkit](https://software.intel.com/en-us/openvino-toolkit) and was tested with the 2020.1 release.

The provided Makefile does the following

1. Downloads the IR files from the [Open Model Zoo](https://github.com/opencv/open_model_zoo)
2. Downloads an aligned face and runs an inference with the age-gender model using the age and gender outputs.

**Note**: The default image used for inference is a CC0 image and was previously cropped/aligned using the face-detection-retail-0004 sample. When using this network, you will have to use a face detection network like face-detection-retail-0004 to align and crop faces to use as input. This will give you the best results.  

## Model Information
### Inputs
 - name: 'input', shape: [1x3x62x62], Expected color order is BGR.
### Outputs 
 - name: 'age_conv3', shape: [1, 1, 1, 1] - Estimated age divided by 100.
 - name: 'prob', shape: [1, 2, 1, 1] - Softmax output across 2 type classes [female, male].

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

