# multistick_cpp: A Movidius Neural Compute Stick example for multiple devices in C++

This directory contains a C++ example that shows how to program for mulitiple NCS devices.  The program opens two NCS devices and uses one device to run GoogLeNet inferences and the other device to run SqueezeNet inferences.

## Prerequisites

This code example requires that the following components are available:
1. Movidius Neural Compute Stick
2. Movidius Neural Compute SDK


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
--- NCS 1 inference ---
Successfully loaded the tensor for image ../../../data/images/nps_electric_guitar.png
Successfully got the inference result for image ../../../data/images/nps_electric_guitar.png
Index of top result is: 546
Probability of top result is: 0.994141
-----------------------

--- NCS 2 inference ---
Successfully loaded the tensor for image ../../../data/images/nps_baseball.png
Successfully got the inference result for image ../../../data/images/nps_baseball.png
Index of top result is: 429
Probability of top result is: 1.000000
-----------------------

~~~



