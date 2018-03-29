# Introduction
The mtcnn network can be used for face detection. The provided Makefile does the following
1. Download a test image
2. Download trained models: one is proposal net with fixed input size 28 * 38, one is output net.
3. Compiles the network using the Neural Compute SDK.
4. There is a python example (run.py) which runs the test image and prints coordinates of detected faces to show how to use the network with the Neural Compute API thats provided in the Neural Compute SDK.

The network is based on the [MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment) . It was modified, simplified and saved in [Movidius Face Detection](https://github.com/ihere1/movidius-face) so that it can be compiled with the NCSDK compiler.

# Makefile
Provided Makefile provides targets to do all of the following.

## make all
Makes, downloads, and prepares everything needed to run the example program.

## make clean
Removes all the temporary and target files that are created by the Makefile.

## make compile
Compiles models to generate Movidius internal 'graph' format files. 'p2838.graph' and 'o.graph' can be loaded on the Neural Compute Stick for face detection.  Demonstrates NCSDK tool: mvNCCompile

## deps 
Downloads the test image and trained models for compilation with the NCSDK.

## make help
Shows makefile possible targets and brief descriptions.

## make run
Runs the provided program which demonstrates using the NCSDK to process the test image using this network.