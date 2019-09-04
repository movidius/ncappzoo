# MobileNets
## Introduction

<a href="https://arxiv.org/abs/1704.04861" target="_blank">MobileNets</a> are a class of efficient convolutional neural networks (CNNs) designed for mobile and embedded vision applications. MobileNets use depth multiplier and image size as hyper-parameters, which can be used to tweak accuracy and latency of the model during training. This ability to tweak the model allows the model builder to train a model that strikes a perfect balance between the application requirements and hardware constrains of their system.

TensorFlow™ provides a set of pre-trained models trained on <a href="http://www.image-net.org/" target="_blank">ImageNet</a>, with different combinations of depth multiplier and image size. The Makefile in this project helps convert these <a href="https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md" target="_blank">TensorFlow MobileNet models</a> to an OpenVINO IR (intermediate representation) file, which can be deployed on to the Intel® Neural Compute Stick 2 (NCS 2) for inference. The <a href="https://movidius.github.io/blog/ncs-rpi3-mobilenets/" _target="blank">NCS developer blog</a> provides some benchmarking numbers related to MobileNets running on NCS and a Raspberry Pi single board computer.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://software.intel.com/en-us/neural-compute-stick/where-to-buy" target="_blank">Intel Neural Compute Stick 2</a>
2. <a href="https://software.intel.com/en-us/openvino-toolkit" target="_blank">OpenVINO R2 toolkit</a>
3. <a href="https://github.com/tensorflow/tensorflow" target="_blank">TensorFlow source repo</a>
4. <a href="https://github.com/tensorflow/models" target="_blank">TensorFlow models repo</a>

The Makefile will clone the TensorFlow source and models repo. 

## Running this Example

~~~
make run
~~~

If `make` ran normally and your computer is able to connect to the NCS device, the output will be similar to this:

~~~
Downloading checkpoint files...
...
Exporting GraphDef file...
...
Freezing model for inference...
...

You should also see a newly created `ncappzoo/tensorflow/mobilenets/model` folder.

## Configuring this example
This example profiles MobileNet V1 with `DEPTH=1.0` and `IMGSIZE=224` by default, but you can profile MobileNet models with other depth and image size settings. Below are some example commands:

Depth multiplier = 0.25 and Image size = 128
~~~
make DEPTH=0.25 IMGSIZE=128
~~~

Depth multiplier = 1.0 and Image size = 192
~~~
make IMGSIZE=192
~~~

### Full list of options
| Depth multiplier |
| --- |
| DEPTH=1.0 |
| DEPTH=0.75 |
| DEPTH=0.50 |
| DEPTH=0.25 |

| Image size |
| --- |
| IMGSIZE=224 |
| IMGSIZE=192 |
| IMGSIZE=160 |
| IMGSIZE=128 |


Compile MobileNet_v1_0.50_128
~~~
make compile_model DEPTH=0.50 IMGSIZE=128
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

