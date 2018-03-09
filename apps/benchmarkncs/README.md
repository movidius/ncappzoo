# benchmarkncs 
A Neural Compute Stick example program that shows how to determine the frames/inferences per second that you are achieving with your current configuration for a specified network and image set.

This directory contains a python example program that will run inferences and report the FPS

## Prerequisites

This code example requires that the following components are available:
1. Movidius Neural Compute Sticks
2. Movidius Neural Compute SDK


## Building and running the example
To run the example code against multiple networks do the following :
1. Open a terminal and change directory to the benchmarkncs base directory
2. Type the following command in the terminal: make run 
The included makefile will build the network graph files if needed as well as run the python program one time for each network

To run the program for a single network do the following:
1. Open a terminal and change directory to the benchmarkncs base directory
2. type the following command  ./benchmarkncs.py <network directory> <image directory> <image width> <image height>
   for example: ./benchmarkncs.py ../../caffe/GoogLeNet ../../data/images 224 224




