# Introduction
The NCS facenet example network is based the work done by David Sandberg here: https://github.com/davidsandberg/facenet.
Facenet is not a classifier that is trained to classify a face as belonging to a particular individual that it was trained on.  Instead it is trained to find and quantify landmarks on faces in general.  By comparing the face landmark quantification values (network inference output) on two images, it is possible to determine how likely the two faces are of the same person.

The provided run.py python program reads a single image file named validated_images/valid.jpg and runs an inference on that "validated" image upon start up.  The results of the valid imge are saved and later used for comparison.  Next each image file in the base directory (the test images) is read one by one.  After an image is read in, it is run through the facenet network and the results are compared with the valid image.  When the network results for the valid image is similar to the results for a test image it is considered a match.  That is to say they the two images are of the same person. 

To determine a match the FACE_MATCH_THRESHOLD value is used.  You might want to adjust this value keeping in mind that the closer this value is to 0.0, the more exact and less forgiving the matching will be.  The initial value for FACE_MATCH_THRESHOLD is 1.2 but you will likely want to play with this value for your own needs.

As the program cycles through the test images in the base directory it will display each image in a GUI window along with a green frame for a match or a red frame indicating no match.  While this window is in focus press any key to advance to the next image.  After all the images have been compared for matches and displayed the program will exit.

The provided Makefile does the following:
1. Downloads test images.
2. Converts unzips, and converts the trained facenet network from https://github.com/davidsandberg/facenet to a format suitable for NCS compilation
3. Compiles the converted network to NCS graph file
4. Runs the provided run.py program that creates a GUI window that shows the test images and match/not match status.

# Prerequisites
This program requires:
- 1 NCS device
- NCSDK 1.12.00 or greater
- The 20170512-110547.zip file from https://github.com/davidsandberg/facenet must be downloaded and copied to the base facenet directory.  Direct google drive link is: https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk .

Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets.

## make all
Builds and/or gathers all the required files needed to run the example run.py, except for downloading 20170512-110547.zip, this must be manually downloaded from here: https://github.com/davidsandberg/facenet and placed in the base directory.

## make model
Unzips the 20170512-110547.zip file, and converts to an NCS-compilable version of the model.  See https://movidius.github.io/ncsdk/tf_compile_guidance.html for general guidance on what changes need to be done to make a TensorFlow model compilable. 

## make compile
Compiles the converted network creating a graph file (facenet_celeb_ncs.graph) for the NCS device.

## make run_py
Runs the provided python program demonstrating the facenet network.

## make clean
Removes all the temporary files that are created by the Makefile.  Also removes 20170512-110547.zip.

