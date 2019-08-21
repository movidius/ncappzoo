# Tiny yolo v3
## Introduction
The TinyYolo network can be used for object recognition and classification. This model was trained with the Coco data set and can detect up to 80 classes. See [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/) for more information on this network. 

The provided Makefile does the following

The provided Makefile does the following
1. Clones a repo that will help to conver the darknet weights to tensorflow. (https://github.com/mystic123/tensorflow-yolo-v3).
2. Download the labels and weights from the [Tiny Yolo v3 site](https://pjreddie.com/darknet/yolo/).
3. Converts the weights and generates a Tensorflow frozen pb file.
4. Compiles the Tensorflow frozen pb file to an IR (intermediate representation) using the Model Optimizer.
4. Runs the provided tiny_yolo_v3.py program that does a single inference on a provided image as an example on how to use the network using the Inference Engine Python API.

## Running this Example
~~~
make run
~~~
**Note**: You can also specify the camera using the command ~~~make run INPUT=cam~~~


## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run or make run_py
Runs a sample application with the network.

### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data, compile_model, compile_cpp.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.


