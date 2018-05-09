# Introduction
The [inception v1](https://github.com/tensorflow/models/tree/master/slim/nets) network can be used for image classification.  The provided Makefile does the following
1. Downloads the TensorFlow checkpoint file
2. Runs the conversion/save python script to generate network.meta file.
3. Profiles, Compiles and Checks the network using the Neural Compute SDK.
4. There is a run.py provided that does a single inference on a provided image as an example on how to use the network using the Neural Compute API

# Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

## make all
Runs ncprofile, ncompile and run.

## make profile
Runs the provided network on the NCS and generates per layer statistics that are helpful for understanding the performance of the network on the Neural Compute Stick.

## make compile
Uses the network description and the trained weights files to generate a Movidius internal 'graph' format file.  This file is later used for loading the network on to the Neural Compute Stick and executing the network.

## make run
Runs the provided run.py file which sends a single image to the Neural Compute Stick and receives and displays the inference results.

## make check
Runs the network on Caffe on CPU and runs the network on the Neural Compute Stick.  Check then compares the two results to make sure they are consistent with each other.

## make clean
Removes all the temporary files that are created by the Makefile

