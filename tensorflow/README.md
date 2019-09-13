# TensorFlow Networks for the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 (or original NCS) with OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit
This directory contains multiple subdirectories. Each subdirectory contains software, data, and instructions that pertain to using a specific TensorFlow neural network with a Neural Compute device such as the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2.  Along with the trained network itself, examples are provided via Makefile that show how the OpenVINO Model Optimizer can be used to compile the network to Intermediate Representation (IR) and also how to create a program that uses that IR model for inferencing.  The sections below are categorized by network type and include a brief explaination of each network.

# TensorFlow Image Classification Networks for Neural Compute devices
|Image Classification Network| Description |
|---------------------|-------------|
|[inception_v1](inception_v1/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v2](inception_v2/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v3](inception_v3/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v4](inception_v4/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[mnist](mnist/README.md) |Network that classifies handwritten digits.  This network is based on  tensorflow mnist_deep.  This project also includes an option to train the network yourself.|
|[mobilenets](mobilenets/README.md) |The mobilenets ([as described in the MobileNets Paper](https://arxiv.org/abs/1704.04861)) are small, low-latency, low-power Convolutional Neural Networks for Mobile Vision Applications.  They are parameterized for a variety of different uses.  Multiple trained networks with different parmameter values are compiled in this directory. |

# TensorFlow Object Detection Networks for Neural Compute devices
|Network| Description |
|---------------------|-------------|
|TBD|TBD|


# TensorFlow Misc Networks for Neural Compute devices
|Network| Description |
|---------------------|-------------|
|TBD|TBD|
