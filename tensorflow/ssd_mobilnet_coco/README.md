# Introduction
The [TensorFlow SSD Mobilenet V1/V2 COCO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) networks can be used for object detection.  The provided Makefile does the following
1. Downloads the TensorFlow SSD Mobilenet network files.
2. Compiles the network using the Neural Compute SDK.
4. There is a run.py provided that does a single inference on a provided image as an example on how to use the network using the Neural Compute API

# Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

## make all
Runs mvNCCompile and make run.

## make compile_v1
Uses the SSD Mobilenet V1 COCO network description and the trained weights files to generate a Movidius internal 'graph' format file.  This file is later used for loading the network on to the Neural Compute Stick and executing the network.

## make compile_v2
Uses the SSD Mobilenet V2 COCO network description and the trained weights files to generate a Movidius internal 'graph' format file.  This file is later used for loading the network on to the Neural Compute Stick and executing the network.

## make run_v1 or make run
Runs the provided run.py file which sends a single image to the Neural Compute Stick and receives and displays the inference results. This command uses the SSD Mobilenet V1 COCO model.

## make run_v2
Runs the provided run.py file which sends a single image to the Neural Compute Stick and receives and displays the inference results. This command uses the SSD Mobilenet V2 COCO model.

## make clean
Removes all the temporary files that are created by the Makefile
