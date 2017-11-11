# Introduction
The birds application is a bird recognition and classification program.  It uses both the NCS along with TinyYolo and GoogLeNet to take an image and first get bounding boxes for birds in the image and then uses GoogLeNet to further classify the birds found.  It depends on the caffe/TinyYolo and caffe/GoogLeNet appzoo examples.
The provided Makefile does the following
1. Builds both TinyYolo and GoogLeNet from their repective directories within the repo
2. Copies build NCS graph files from TinyYolo and GoogLeNet to the current directory 
3. Runs the provided birds.py program that creates a GUI window that shows all the birds and all their classifications using the network using the Neural Compute API

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets

## make all
builds and/or gathers all the required files needed to run the application


## make run
Runs the provided birds.py python program which does the bird classification

## make clean
Removes all the temporary files that are created by the Makefile
