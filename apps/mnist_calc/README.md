# Introduction
This project uses MNIST to do digit classification for a touchscreen handwriting-input calculator.

The provided Makefile does the following:
1. Builds tensorflow MNIST graph file from the tensorflow/mnist directory in the repository.
2. Copies the built NCS graph file from the mnist directory to the project base directory.
3. Runs touchcalc.py, which creates a GUI window for calculator input and handles calculator control. 

# Prerequisites
All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

This program requires:
- 1 NCS device
- NCSDK 2.04 or greater
- OpenCV with associated Python bindings*. 

*The ncsdk includes an install-opencv.sh script to handle basic installation.

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets.

## make all
Builds and/or gathers all the required files needed to run the application.

## make run
Runs the touchcalc application.

## make clean
Removes all the temporary files that are created by the Makefile.

