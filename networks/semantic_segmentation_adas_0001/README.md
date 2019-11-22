# semantic segmentation adas 0001
## Introduction
This app does semantic segmentation using the [semantic-segmentation-adas-0001](https://docs.openvinotoolkit.org/2019_R1/_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html), the [Intel Movidius Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick), and the [OpenVINO Toolkit R3](https://software.intel.com/en-us/openvino-toolkit). Each of the colors represents a class. The classes that this network can detect are:

- road
- sidewalk
- building
- wall
- fence
- pole
- traffic light
- traffic sign
- vegetation
- terrain
- sky
- person
- rider
- car
- truck
- bus
- train
- motorcycle
- bicycle
- ego-vehicle

## Prerequisites
This program requires:
- 1 NCS2/NCS1 device
- OpenVINO 2019 R3 Toolkit

Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.


## Building the Example

To run the example code do the following :
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make all```

## Running the Example

After building the example you can run the example code by doing the following :
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make run``` 

When the application runs normally, another window should pop up and show the semantic segmentation colors overlayed onto the sample image.

**Keybindings**:
- q or Q - Quit the application

## Model Information
### Inputs
 - name: 'data', shape: [1x3x1024x2048], Expected color order is BGR after optimization. 
### Outputs 
 - name: 'argmax', shape: [1x3x1024x2048] - Each value represents a pixel in an image and the index of a class.

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

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 

### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.
 
### make clean
Removes all the temporary files that are created by the Makefile.

