# tiny yolo v3
## Introduction
The tiny yolo v3 network can be used for object recognition and classification. This model was trained with the Coco data set and can detect up to 80 classes. See [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/) for more information on this network. 

This sample utilizes the OpenVINO Inference Engine from the [OpenVINO Deep Learning Development Toolkit](https://software.intel.com/en-us/openvino-toolkit) and was tested with the 2019 R2 release.

The provided Makefile does the following:
1. Clones a [Tensorflow yolo repo](https://github.com/mystic123/tensorflow-yolo-v3) that will help to convert the darknet weights to tensorflow.
2. Download the labels and weights from the [Tiny Yolo v3 site](https://pjreddie.com/darknet/yolo/).
3. Converts the weights and generates a Tensorflow frozen pb file.
4. Compiles the Tensorflow frozen pb file to an IR (intermediate representation) using the Model Optimizer.
4. Runs the provided tiny_yolo_v3.py program that does a single inference on a provided image as an example on how to use the network using the Inference Engine Python API.

## Model Information
### Inputs
 - name: 'data', shape: [1x3x416x416], Expected color order is RGB.
### Outputs 
 - name: 'detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion', shape: [1, 255, 26, 26].
 - name: 'detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion', shape: [1, 255, 13, 13].

**Note**: The '26' and '13' values in the output represent the number of grid cells for each output. 

## Running this Example
~~~
make run
~~~
**Note**: You can also specify the camera using the command: ```make run INPUT=cam```
**Note**: You may need to adjust some values to better detect objects 

## Algorithm Thresholds
There are a few thresholds in the code you may want to tweek if you aren't getting results that you expect:
- <strong>DETECTION_THRESHOLD</strong>: This is the minimum probability allowed for boxes returned from tiny yolo.  This should be between 0.0 and 1.0.  A lower value will allow more boxes to be displayed.
- <strong>IOU_THRESHOLD</strong>: Determines which boxes from Tiny Yolo should be separate objects vs identifying the same object.  This is based on the intersection-over-union calculation.  The closer this is to 1.0 the more similar the boxes need to be in order to be considered around the same object.

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


