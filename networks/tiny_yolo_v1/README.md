# tiny_yolo_v1
## Introduction
The [Tiny Yolo v1](https://pjreddie.com/darknet/yolov1/) network can be used for object recognition and classification. This model was trained with the Pasval VOC data set and can detect up to 20 classes. See [https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/) for more information on this network. 

This sample utilizes the OpenVINO Inference Engine from the [OpenVINO Deep Learning Development Toolkit](https://software.intel.com/en-us/openvino-toolkit) and was tested with the 2020.1 release.

The provided Makefile does the following

The provided Makefile does the following
1. Downloads the Caffe prototxt file 
3. Downloads the .caffemodel weights file.
3. Compiles the IR (intermediate representation) files using the Model Optimizer.
4. Runs the provided run.py program that does a single inference on a provided image as an example on how to use the network using the Inference Engine Python API.


## Model Information
### Inputs
 - name: 'data', shape: [1x3x448x448], Expected color order is BGR. Original network expects RGB, but for this sample, the IR is compiled with the --reverse_input_channels option to convert the IR to expect the BGR color order.
### Outputs 
 - name: 'prob', shape: [1, 1470].

The model splits the image into a 7x7 grid (49 grid cells total). For each grid cell, there are 2 anchor boxes. Each of these two anchor boxes have a object score and 20 class probability values. An objects "final score" is calculated by multiplying the object score by the class probability. 

Post processing for this model includes going through all "final scores" and filtering out low scoring objects. After filtering out low scoring objects, duplicate objects will need to be filtered via intersection over union (iou) calculations. 
 
  
**Note**: The output (1470 element array) for this model is interpreted as follows:
 - Elements (0-979) represent the 980 class probabilities (49 grid cells x 20 classes = 980 total class probabilities)
 - Elements (980-1077) represent the 98 object scores (49 grid cells x 2 anchor boxes = 98 total object scores). These scores represent how likely the box contains an object.
 - Elements (1078-1469) represent the 392 bounding box coordinates (98 anchor boxes x 4 bounding box coordinates = 392 total bounding box coordinates). 


## Algorithm Thresholds
There are a few thresholds in the code you may want to tweek if you aren't getting results that you expect:
- <strong>DETECTION_THRESHOLD</strong>: This is the minimum probability allowed for boxes returned from tiny yolo v1.  This should be between 0.0 and 1.0.  A lower value will allow more boxes to be displayed.
- <strong>IOU_THRESHOLD</strong>: Determines which boxes from Tiny Yolo v1 should be separate objects vs identifying the same object.  This is based on the intersection-over-union calculation.  The closer this is to 1.0 the more similar the boxes need to be in order to be considered around the same object.

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
Makes the follow items: deps, data, compile_model.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.



