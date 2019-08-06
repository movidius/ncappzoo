# benchmark_ncs 
## Introduction
This example program that shows how to determine the fps (frames per second or inferences per second) that you are achieving with your current configuration for a specified network and image set. 

The example uses 3 threads per NCS device and by default creates 6 async inference requests per thread. By default, the sample will run 1k inferences using [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).

The provided Makefile does the following:
1. Builds the IR files using the model files from [Open Model Zoo](https://github.com/opencv/open_model_zoo).
2. Copies the IR files from the model directory to the project base directory.
3. Runs the sample.

## Running the Example
To run the example code do the following :
1. Open a terminal and change directory to the project base directory
2. Type the following command in the terminal: ```make run``` 

To get a list of commandline options type the following command: ```python3 benchmark_ncs.py help```

**Note**: The CPU device can be used with the example using the command: ```make run DEV=CPU```

**Note**: Other models can be used with the example.  

*Example of running benchmark_ncs on AlexNet:* ```make run XML=../../caffe/AlexNet/alexnet.xml BIN=../../caffe/AlexNet/alexnet.bin```

## Prerequisites
This code example requires that the following components are available:
1. 1 or more NCS devices
2. OpenVINO 2019 R2 Toolkit

*It may run with older versions but you may see some glitches such as the GUI Window not closing when you click the X in the title bar, and other key binding issues.

Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

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
 
### make clean
Removes all the temporary files that are created by the Makefile.

