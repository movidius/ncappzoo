# benchmark_ncs 
## Introduction
This example program that shows how to determine the fps (frames per second or inferences per second) that you are achieving with your current configuration for a specified network and image set. 

The example uses 3 threads per NCS device and creates 6 async inference requests per thread by default. The sample will also run 1k inferences using [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) by default.

The provided Makefile does the following:
1. Builds the IR files for the default model.  Note: To run on other networks use the commandline make arguments XML and BIN to specify an IR model that has already been built.
2. Copies the IR files from the model directory to the project base directory.  
3. Runs the sample.

## Running the Example
To run the example code with default settings do the following :
1. Open a terminal and change directory to the project base directory
2. Type the following command in the terminal: ```make run``` 

The benchmark_ncs.py program itself has some options that aren't accessible via the make file.  To get a list of commandline options type the following command: ```python3 benchmark_ncs.py help```.  Then you can try the different options listed in the help by invoking the program in the same way (directly through the python interpreter).

**Note**: The CPU device can be used with the example using the command: ```make run DEV=CPU```

**Note**: Other models can be used with the example by using the XML and BIN commandline options.  

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

### make default_model
Compiles an IR file from a default model to be used when running the sample.

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.
 
### make clean
Removes all the temporary files that are created by the Makefile.

