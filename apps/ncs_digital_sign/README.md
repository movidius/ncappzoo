# ncs_digital_sign: 
## Introduction
This digital signage app dynamically outputs advertisements based on facial detection and age/gender inference using the Intel Neural Compute Stick 2. 

The example does face detection on a camera frame using face-detection-retail.0004, crops the detected faces, then does age and gender inference using the age-gender network. The example outputs different advertisements based on user's demographics using the age-gender-recognition model. All models can be found on the [Open Model Zoo](https://github.com/opencv/open_model_zoo). This sample uses pre-compiled IRs, so the model optimizer is not utilized.

![](updated_gif.gif)

Images in the advertisement_images directory are subject to the licenses.txt file within the directory.

## Building the Example

To run the example code do the following :
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make all```

## Running the Example

After building the example you can run the example code by doing the following :
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make run``` 

When the application runs normally, another window should pop up and show the feed from the webcam/usb cam. The program should perform inferences on faces on frames taken from the webcam/usb cam.

## Prerequisites
This program requires:
- 1 NCS1/NCS2 device
- OpenVINO 2020.1 Toolkit
- A webcam (laptop or USB)


Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine. ARM devices may be supported, but has not been verified.

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make run
Runs the sample application.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application.

### make data
Gathers all of the required data need to run the sample.

### make deps
Builds all of the dependencies needed to run the sample.

### make install_reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 

### make uninstall_reqs
Checks required packages that are able to be uninstalled. 
 
### make clean
Removes all the temporary files that are created by the Makefile.


