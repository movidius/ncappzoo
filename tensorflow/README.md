# TensorFlow Networks for the NCSDK
This directory contains multiple subdirectories. Each subdirectory contains software, data, and instructions that pertain to using a specific TensorFlow neural network with a Neural Compute device such as the Intel Neural Compute Stick.  Typically examples are provided that show how the NCSDK can be used compile the network to a graph file and also how to create a program that uses that graph file for inferencing.  The sections below are categorized by network type and include a brief explaination of each network.

# TensorFlow Image Classification Networks for NCSDK
|Image Classification Network| Description |
|---------------------|-------------|
|[inception_v1](inception_v1/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v2](inception_v2/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v3](inception_v3/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v4](inception_v4/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[mnist](mnist/README.md) |Network that classifies handwritten digits.  This network is based on  tensorflow mnist_deep as outlined in [NCSDK TensorFlow Guidance page. ](https://movidius.github.io/ncsdk/tf_compile_guidance.html)  This project also includes an option to train the network yourself.|
|[mobilenets](mobilenets/README.md) |The mobilenets ([as described in the MobileNets Paper](https://arxiv.org/abs/1704.04861)) are small, low-latency, low-power Convolutional Neural Networks for Mobile Vision Applications.  They are parameterized for a variety of different uses.  Multiple trained networks with different parmameter values are compiled in this directory. |

# TensorFlow Misc Networks for NCSDK
|Network| Description |
|---------------------|-------------|
|[facenet](facenet/README.md) |FaceNet is a nework that is trained to find and quantify landmarks on faces in general.  By comparing the face landmark quantification values (network inference output) on two images, it is possible to determine how likely the two faces are of the same person.  This is based on [work by David Sandberg](https://github.com/davidsandberg/facenet).  |
