# Introduction
The TinyYolo network can be used for object recognition and classification.  See [https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/) for more information on this network. 
The provided Makefile does the following
1. Downloads the Caffe prototxt file 
3. Downloads the .caffemodel file which was trained.
3. Profiles and Compiles the network using the Neural Compute SDK.
4. Runs the provided run.py program that does a single inference on a provided image as an example on how to use the network using the Neural Compute API

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets

## make all
Runs profile, compile.

## make profile
Runs the provided network on the NCS and generates per layer statistics that are helpful for understanding the performance of the network on the Neural Compute Stick.

## make compile
Uses the network description and the trained weights files to generate a Movidius internal 'graph' format file.  This file is later used for loading the network on to the Neural Compute Stick and executing the network.

## make run_py
Runs the provided run.py python program which sends a single image to the Neural Compute Stick and receives and displays the inference results along with a GUI window showing the identified objects in the image.

## make clean
Removes all the temporary files that are created by the Makefile
