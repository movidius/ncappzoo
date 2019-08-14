# face-detection-retail-0004
## Introduction
The [face-detection-retail-0004](https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-retail-0004/description/face-detection-retail-0004.md) network can be used for face detection. 

The provided Makefile does the following

1. Downloads the IR files from the [Open Model Zoo](https://github.com/opencv/open_model_zoo)
2. Takes an image and runs an inference on the face-detection-retail-0004 model.

The sample can also be used to crop images and write them to file. 

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
Makes the follow items: deps, data.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation.
 
### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.

### make clean
Removes all the temporary and target files that are created by the Makefile.

