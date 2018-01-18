# Introduction
The video_face_matcher example app uses the TensorFlow [ncappzoo/tensorflow/facenet](../../tensorflow/facenet) neural network to find a face in a video camera stream that matches with a known face image.  

The provided video_face_matcher.py python program starts a webcam and shows the camera preview in a GUI window.  When the camera preview shows a face that matches the known valid face (video_face_matcher/validated_images/valid.jpg) a green frame is displayed around the window to indicate a match was found.   

# Details
To make the program recognize your face or any face that you would like replace the video_face_matcher/validated_images/valid.jpg image file with a similar image of the person you would like to recognize.

To determine a match the FACE_MATCH_THRESHOLD value is used.  You might want to adjust this value keeping in mind that the closer this value is to 0.0, the more exact and less forgiving the matching will be.  The initial value for FACE_MATCH_THRESHOLD is 1.2 but you will likely want to play with this value for your own needs.

The provided Makefile does the following:
1. Makes the facenet graph file from the ncappzoo/tensorflow/facenet example and copies it to the base directory of this project.
2. Runs the provided video_face_matcher.py program that creates a GUI window which shows the camera stream and match/not match status.

# Prerequisites
This program requires:
- 1 NCS device
- A webcam
- NCSDK 1.12.00 or greater
- The ncappzoo/tensorflow/facenet make compile command must work.  See [the facenet README.md](../../tensorflow/facenet/README.md) for details
- opencv 3.3 with video for linux support

Note: The OpenCV version that installs with the some versions of ncsdk does <strong>not</strong> provide V4L support.  To run this application you will need to replace the ncsdk version with a version built from source.  To remove the old opencv and build and install a compatible version you can run the following command from the app's base directory:

```
   make opencv
```   
Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets.

## make all
Builds and/or gathers all the required files needed to run the example python program. 

## make compile
runs make compile in the ncappzoo/tensorflow/facenet project to create the facenet graph file (facenet_celeb_ncs.graph) for the NCS device, and copies it to the base directory.

## make run_py
Runs the provided python program demonstrating the facenet network.

## make clean
Removes all the temporary files that are created by the Makefile.  Also removes 20170512-110547.zip.

