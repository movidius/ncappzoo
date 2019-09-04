# Inception_v1
## Introduction

<a href="https://research.google.com/pubs/pub43022.html" target="_blank">Inception</a> is a deep convolutional neural network (CNN) architecture designed by Google during the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC2014). The main goal behind this model is to improve accuracy by increasing depth and width of the network without affecting the computational requirements. However, the latency of inception based models like GoogLeNet, Inception V1, V2, V3 and V4 is much larger than that of MobileNets. 

TensorFlow™ provides different versions of pre-trained inception models trained on <a href="http://www.image-net.org/" target="_blank">ImageNet</a>. The Makefile in this project helps convert these <a href="https://github.com/tensorflow/models/tree/master/research/slim#Pretrained" target="_blank">TensorFlow Inception models</a> to an IR format file (Intermediate Representation), which can be deployed on to the Intel® Neural Compute Stick (NCS1/NCS2) for inference.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://software.intel.com/en-us/neural-compute-stick/where-to-buy" target="_blank">Intel Neural Compute Stick</a>
2. <a href="https://software.intel.com/en-us/openvino-toolkit" target="_blank">Intel OpenVINO 2019 R2 Toolkit</a>
3. <a href="https://github.com/tensorflow/tensorflow" target="_blank">TensorFlow source repo</a>
4. <a href="https://github.com/tensorflow/models" target="_blank">TensorFlow models repo</a>

## Running this Example
You can run the sample with the command:
~~~
make run
~~~

## Compiling the models
You can compile the model to an IR using the command:
~~~
make compile_model 
~~~

## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run or make run_py
Runs a sample application with the network.

### make cpp
Builds the C++ example program run_cpp which can be executed with make run_cpp. 

### make run_cpp
Runs the provided run_cpp executable program that is built via make cpp.  This program sends a single image to the Neural Compute Stick and receives and displays the inference results.

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

## Troubleshooting

~~~
Makefile:31: *** TF_MODELS_PATH is not defined. Run `export TF_MODELS_PATH=path/to/your/tensorflow/models/repo`.  Stop.
~~~
* Make sure TF_MODELS_PATH is pointing to your tensorflow models directory.

~~~
Makefile:46: *** TF_SRC_PATH is not defined. Run `export TF_SRC_PATH=path/to/your/tensorflow/source/repo`.  Stop.
~~~
* Make sure TF_SRC_PATH is pointing to your tensorflow source directory.

