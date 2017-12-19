# Introduction
This project uses SSD MobileNet to do object recognition and classification for a street camera. Rather than a camera, video files will be used to simulate a camera. 

The provided Makefile does the following:
1. Builds both caffe ssd mobilenet graph file from the caffe/SSD_MobileNet directory in the repository.
2. Copies the built NCS graph file from the SSD_MobileNet directory to the project base directory
3. Downloads some sample traffic video files.
3. Runs the provided street_cam_ssd_mobilenet.py program which creates a GUI window that shows the video stream along with labels and boxes around the identified objects. 

# Prerequisites
This program requires:
- 1 NCS device
- NCSDK 1.11 or greater
- opencv 3.3 with video for linux support

Note: The OpenCV version that installs with the current ncsdk (1.10.00) does <strong>not</strong> provide V4L support.  To run this application you will need to replace the ncsdk version with a version built from source.  To remove the old opencv and build and install a compatible version you can run the following command from the app's base directory:
```
   make opencv
```   
Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.


# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets.

## make all
Builds and/or gathers all the required files needed to run the application except building and installing opencv (this must be done as a separate step with 'make opencv'.)

## make videos
Downloads example video files.

## make opencv
Removes the version of OpenCV that was installed with the NCSDK and builds and installs a compatible version of OpenCV 3.3 for this app. This will take a while to finish. Once you have done this on your system you shouldn't need to do it again.

## make run_py
Runs the provided python program which shows the video stream along with the object boxes and classifications.

## make clean
Removes all the temporary files that are created by the Makefile
