# facenet
## Introduction
The OpenVINO Ncappzoo facenet example network is based the work done by David Sandberg here: https://github.com/davidsandberg/facenet.
Facenet is not a classifier that is trained to classify a face as belonging to a particular individual that it was trained on.  Instead it is trained to find and quantify landmarks on faces in general.  By comparing the face landmark quantification values (network inference output) on two images, it is possible to determine how likely the two faces are of the same person.

By default, the sample uses aligned face images that are cropped using the face-detection-retail-0004 sample. If you use this network, make sure to use a face detection network like face-detection-retail-0004 to align/crop your images for the best results. 

The application then reads a single image file named validated_face/valid_face.png and runs an inference on that "validated" image upon start up.  The results of the valid image are saved and later used for comparison.  Next each image file in the test_faces directory (the test images) is read one by one.  After an image is read in, it is run through the facenet network and the results are compared with the valid image.  When the network results for the valid image is similar to the results for a test image it is considered a match.  That is to say they the two images are of the same person. 

To determine a match the FACE_MATCH_THRESHOLD value is used.  You might want to adjust this value keeping in mind that the closer this value is to 0.0, the more exact and less forgiving the matching will be.  The initial value for FACE_MATCH_THRESHOLD is 0.91 but you will likely want to play with this value for your own needs.

As the program cycles through the test images in the base directory it will display each image in a GUI window along with a green frame for a match or a red frame indicating no match.  While this window is in focus press any key to advance to the next image.  After all the images have been compared for matches and displayed the program will exit.

This sample utilizes the OpenVINO Inference Engine from the [OpenVINO Deep Learning Development Toolkit](https://software.intel.com/en-us/openvino-toolkit) and was tested with the 2020.1 release.

The provided Makefile does the following:
1. Downloads test images.
2. Converts unzips, and converts the trained facenet network from https://github.com/davidsandberg/facenet to a format suitable for NCS compilation
3. Compiles the converted network to NCS graph file
4. Crops all images and copies them to the appropriate folder.
4. Runs the provided facenet.py program that creates a GUI window that shows the test images and match/not match status.

## Prerequisites
This program requires:
- 1 NCS2/NCS1 device
- OpenVINO 2020.1
- The 20170512-110547.zip file from https://github.com/davidsandberg/facenet must be downloaded and copied to the base facenet directory.  Direct google drive link is: https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk .

**Note**: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

## Model Information
### Inputs
 - name: 'image_batch/placeholder_port_0', shape: [1x3x160x160], Expected color order is BGR after optimization. Original network expects RGB, but for this sample, the IR is compiled with the --reverse_input_channels option to convert the IR to expect the BGR color order. 
### Outputs 
 - name: 'InceptionResnetV1/Bottleneck/BatchNorm/Reshape_1/Normalize', shape: [1, 512] - row-vector of 512 floating-point values - face embeddings.


## Running this Example
~~~
make run
~~~

## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run or make run_py
Runs a sample application with the network.

### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data, compile_model.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  
**Note** The model will take a bit of time to compile to an IR format.

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.

