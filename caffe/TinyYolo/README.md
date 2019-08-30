# Tiny yolo v1
## Introduction
The TinyYolo network can be used for object recognition and classification. This model was trained with the Pasval VOC data set and can detect up to 20 classes. See [https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/) for more information on this network. 

The provided Makefile does the following

The provided Makefile does the following
1. Downloads the Caffe prototxt file 
3. Downloads the .caffemodel weights file.
3. Compiles the IR (intermediate representation) files using the Model Optimizer.
4. Runs the provided run.py program that does a single inference on a provided image as an example on how to use the network using the Inference Engine Python API.

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



