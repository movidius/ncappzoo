# Caffe Networks for the NCSDK
This directory contains multiple subdirectories. Each subdirectory contains software, data, and instructions that pertain to using a specific Caffe neural network with a Neural Compute device such as the Intel Neural Compute Stick.  Typically examples are provided that show how the NCSDK can be used compile the network to a graph file and also how to create a program that uses that graph file for inferencing.  The sections below are categorized by network type and include a brief explaination of each network.

# Caffe Image Classification Networks for NCSDK
|Image Classification Network| Description |
|---------------------|-------------|
|[AgeNet](AgeNet/README.md) |Network that classifies a face image into age ranges. |
|[AlexNet](AlexNet/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[GenderNet](GenderNet/README.md) |Network that classifies a face image as male or female. |
|[GoogLeNet](GoogLeNet/README.md) |BAIR/BLVC GoogleNet is a network based on [GoogleNet](https://arxiv.org/abs/1409.4842), the winner of ILSVRC 2014, that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[ResNet-18](ResNet-18/README.md) |Deep Residual network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[SqueezeNet](SqueezeNet/README.md) |Accuracy similar to AlexNet with many fewer parameters and small model size as described int the [SqueezeNet paper](https://arxiv.org/abs/1602.07360). Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |

# Caffe Object Detection Networks for NCSDK
|Object Detection Network| Description |
|---------------------|-------------|
|[SSD_MobileNet](SSD_MobileNet/README.md) |MobileNet Single Shot Detector takes an image, detects the 20 PASCAL object classes as specified in the ([Visual Object Classes Challenges](http://host.robots.ox.ac.uk/pascal/VOC/)), their bounding boxes, and classifications.  |
|[TinyYolo](TinyYolo/README.md) |This Tiny You Only Look Once model is based on [tiny-yolo DarkNet model ](https://pjreddie.com/darknet/yolov1/).  Given an image, detects the 20 PASCAL object classes as specified in the ([Visual Object Classes Challenges](http://host.robots.ox.ac.uk/pascal/VOC/)), their bounding boxes, and classifications.  Requires some post processing of results to narrow down relevant boxes.  |
