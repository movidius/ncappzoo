# emotions-recognition-retail-0003
## Introduction
The [emotions-recognition-retail-0003](https://github.com/movidius/ncappzoo/blob/master/networks/emotions-recognition-retail-0003/README.md) network can be used for emotion recognition. This model can be used to align faces for use with face recognition.

The provided Makefile does the following

1. Downloads the IR files from the [Open Model Zoo](https://github.com/opencv/open_model_zoo)
2. Takes an image and runs an inference on the emotions-recognition-retail-0003 model.

The sample can also be used to crop images and write them to file. 

## Model Information
### Inputs
 - name: "data" , shape: [1x3x64x64] - An input image in [1xCxHxW] format. Expected color order is BGR.
### Outputs 
 - name: "prob_emotion", shape: [1, 5, 1, 1] - Softmax output across five emotions ('neutral', 'happy', 'sad', 'surprise', 'anger').

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

