# Introduction
The mnist network can be used for handwriting recognition for the digits 0-9.  The provided Makefile does the following
1. Downloads a trained model 
2. Downloads test images
3. Compiles the network using the Neural Compute SDK.
4. There is a python example (run.py) which runs an inference for all of the test images to show how to use the network with the Neural Compute API thats provided in the Neural Compute SDK.

This network is based on the [TensorFlow 1.4 mnist_deep.py example.](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py)  It was modified, trained, and saved in accordance with the [NCSDK TensorFlow Guidance page ](https://movidius.github.io/ncsdk/tf_compile_guidance.html) so that it can be compiled with the NCSDK compiler.

# Makefile
Provided Makefile provides targets to do all of the following.

## make all
Makes, downloads, and prepares everything needed to run the example program.

## make clean
Removes all the temporary and target files that are created by the Makefile.

## make compile
Compiles the trained model to generate a Movidius internal 'graph' format file.  This file can be loaded on the Neural Compute Stick for inferencing.  Demonstrates NCSDK tool: mvNCCompile

## deps 
Downloads and prepares a trained network for compilation with the NCSDK

## make help
Shows makefile possible targets and brief descriptions. 

## make model
Downloads the trained model

## make run
Runs the provided program which demonstrates using the NCSDK to run an inference using this network.

## make train
Creates an NCSDK compatibile version of the network and trains it.  This may take 20 min or more depending on your system setup and performance.  

