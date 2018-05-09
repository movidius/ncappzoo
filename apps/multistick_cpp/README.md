# multistick_cpp: A Movidius Neural Compute Stick example for multiple devices in C++

This directory contains a C++ example that shows how to program for mulitiple NCS devices.  The program opens two NCS devices and uses one device to run GoogLeNet inferences and the other device to run SqueezeNet inferences.

## Prerequisites

This code example requires that the following components are available:
1. Two Movidius Neural Compute Stick
2. Movidius Neural Compute SDK 2.x


## Building the example
To run the example code do the following :
1. Open a terminal and change directory to the multistick_cpp example base directory
2. Type the following command in the terminal: make


## Running the Example
After building the example you can run the example code by doing the following :
1. Open a terminal and change directory to the multistick_cpp base directory
2. Type the following command in the terminal: make run 

When the application runs normally and is able to connect to the NCS device the output will be similar to this:

~~~
Successfully opened NCS device at index 0!
Successfully opened NCS device at index 1!
Successfully allocated graph and FIFOs for googlenet.graph
Successfully allocated graph and FIFOs for squeezenet.graph

--- NCS 1 inference ---
Successfully queued inference with FIFO elements OK!
Index of top result is: 546
Probability of top result is: 0.994141
-----------------------

--- NCS 2 inference ---
Successfully queued inference with FIFO elements OK!
Index of top result is: 546
Probability of top result is: 0.969727
-----------------------


~~~



