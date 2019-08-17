# AlexNet
## Introduction
The [AlexNet](https://github.com/opencv/open_model_zoo/blob/master/models/public/alexnet/alexnet.md) network can be used for image classification. The model was sourced from the [Open Model Zoo](https://github.com/opencv/open_model_zoo).

The provided Makefile does the following

1. Downloads the Caffe prototxt file and makes any changes necessary to work with OpenVINO (tested with 2019 R2) and the Intel Neural Compute Stick (NCS1/NCS2). 
2. Downloads and generates the required ilsvrc12 data.
3. Downloads the .caffemodel file from [Open Model Zoo](https://github.com/opencv/open_model_zoo).
4. Compiles the network into an IR (intermediate representation) format file (.xml) using the OpenVINO Model Optimizer. 
5. Runs a single inference on an image as an example of how to use this network with the OpenVINO Python API through the simple_classifier_py app.

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
