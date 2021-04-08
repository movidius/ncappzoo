# Face Emotion Game
## Introduction
This app does facial detection and emotion detection using the Intel Movidius Neural Compute Stick 2. 

The example does face detection on a camera frame using face-detection-retail.0004, crops the detected faces, then does emotion recognition using the emotions-recognition-retail-0003 network. When running, the app shows the realtime camera preview while overlaying, a box around faces (color coded for gender), and the facial expressions label. User should do their expression to match the emoji that appears on top of the camera window. All models can be found on the [Open Model Zoo](https://github.com/opencv/open_model_zoo). This sample uses pre-compiled IRs, so the model optimizer is not utilized.

![](src\images\face_emotion_game.png)


## Building the Example

To run the example code do the following:
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make all```

## Running the Example

After building the example you can run the example code by doing the following:
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make run``` 

When the application runs normally, another window should pop up and show the feed from the webcam/usb cam. The program should perform inferences on faces on frames taken from the webcam/usb cam.

## Prerequisites
This program requires:
- 1 x NCS2 device
- 1 x Raspberry Pi 3
- 1 x Webcam (USB)
- OpenVINO 2019 R2 Toolkit

*It may run with older versions but you may see some glitches such as the GUI Window not closing when you click the X in the title bar, and other key binding issues.

Note: All development and testing has been done on Raspberry Pi 3.

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

### make default_model
Compiles an IR file from a default model to be used when running the sample.

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 

### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.
 
### make clean
Removes all the temporary files that are created by the Makefile.


