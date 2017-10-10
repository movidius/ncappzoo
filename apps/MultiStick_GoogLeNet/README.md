# MultiStick_GoogLeNet 
A Neural Compute Stick example program that shows how to use multiple NCS devices to achieve faster inferences. 

This directory contains a python GUI example program that demonstrates the difference between using multiple NCS devices vs a single device.

## Prerequisites

This code example requires that the following components are available:
1. three or more Movidius Neural Compute Sticks
2. Movidius Neural Compute SDK


## Building and running the example
To run the example code do the following :
1. Open a terminal and change directory to the MultiStick_GoogLeNet base directory
2. Type the following command in the terminal: make run 
The included makefile will build the GoogLeNet graph file if needed as well as run the python program

When the application runs normally two GUI windows will appear and they will display images from the ../../data/images directory as well as the inference results for that image.  One of the windows will show results from a single NCS device and the other window will show results from the other NCS devices.  The window showing results from multiple sticks should run faster than the other window.




