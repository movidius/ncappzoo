# Introduction
The [SqueezeNet V1.0](https://github.com/DeepScale/SqueezeNet) network can be used for image classification.  The provided Makefile does the following
1. Downloads the Caffe prototxt file and makes any changes necessary to work with the Movidius Neural Compute SDK
2. Downloads and generates the required ilsvrc12 data
3. Downloads the .caffemodel file which was trained and provided by BVLC.
3. Profiles, Compiles and Checks the network using the Neural Compute SDK.
4. There is a python example (run.py) and a C++ example (cpp/run.cpp) which both do a single inference on an image as an example of how to use this network with the Neural Compute API thats provided in the Neural Compute SDK.

# Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

## make help
Shows makefile possible targets and brief descriptions. 

## make all
Makes the following: prototxt, caffemodel, profile, compile, check, cpp, run, run_cpp.

## make prototxt
Downloads the Caffe prototxt file and makes a few changes necessary to work with the Movidius Neural Compute SDK.

## make caffemodel
Downloads the Caffe model file

## make profile
Runs the provided network on the NCS and generates per layer statistics that are helpful for understanding the performance of the network on the Neural Compute Stick.  Output diplayed on terminal and the output_report.html file is also created.  Demonstrates NCSDK tool: mvNCProfile 

## make browse_profile
profiles the network similar to make profile and then brings up output_report.html in a browser.  Demonstrates NCSDK tool: mvNCProfile 

## make compile
Uses the network description and the trained weights files to generate a Movidius internal 'graph' format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  Demonstrates NCSDK tool: mvNCCompile

## make check
Runs the network on Caffe on the CPU and compares results when run on the Neural Compute Stick.  Consistency results are output to the terminal.  Demonstrates the NCSDK tool: mvNCCheck.

## make run_py
Runs the provided run.py python script which sends a single image to the Neural Compute Stick and receives and displays the inference results.

## make cpp
Builds the C++ example program run_cpp which can be executed with make run_cpp. 

## make run_cpp
Runs the provided run_cpp executable program that is built via make cpp.  This program sends a single image to the Neural Compute Stick and receives and displays the inference results.

## make clean
Removes all the temporary and target files that are created by the Makefile.
