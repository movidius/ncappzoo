# Caffe Networks for the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 (or original NCS) with OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit
This directory contains multiple subdirectories. Each subdirectory contains software, data, and instructions that pertain to using a specific Caffe neural network with a Neural Compute device such as the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2.  Along with the trained network itself, examples are provided via Makefile that show how the OpenVINO Model Optimizer can be used to compile the network to Intermediate Representation (IR) and also how to create a program that uses that IR model for inferencing.  The sections below are categorized by network type and include a brief explaination of each network.

# Caffe Image Classification Networks for Neural Compute devices
|Image Classification Network| Description |
|---------------------|-------------|
|[AgeGenderNet](AgeNet/README.md) |Network that classifies a face image into age ranges. |
|[AlexNet](AlexNet/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[GoogLeNet](GoogLeNet/README.md) |BAIR/BLVC GoogleNet is a network based on [GoogleNet](https://arxiv.org/abs/1409.4842), the winner of ILSVRC 2014, that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[ResNet-50](ResNet-50/README.md) |[Deep Residual network](https://arxiv.org/pdf/1512.03385.pdf)  with 50 layers that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[SqueezeNet](SqueezeNet/README.md) |Accuracy similar to AlexNet with many fewer parameters and small model size as described int the [SqueezeNet paper](https://arxiv.org/abs/1602.07360). Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |

# Caffe Object Detection Networks for Neural Compute devices
|Object Detection Network| Description |
|---------------------|-------------|
|[face-detection-retail-0004](face-detection-retail-0004/README.md) |This is a nework that is trained to find faces in general.  [More information specific to this network is available.](https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-retail-0004/description/face-detection-retail-0004.md) |
|[SSD_Mobilenet](SSD_Mobilenet/README.md)|MobileNet Single Shot Detector takes an image, detects the 20 PASCAL object classes as specified in the ([Visual Object Classes Challenges](http://host.robots.ox.ac.uk/pascal/VOC/)), their bounding boxes, and classifications. |
|[TinyYolo](TinyYolo/README.md) |This Tiny You Only Look Once model is based on [tiny-yolo DarkNet model ](https://pjreddie.com/darknet/yolov1/).  Given an image, detects the 20 PASCAL object classes as specified in the ([Visual Object Classes Challenges](http://host.robots.ox.ac.uk/pascal/VOC/)), their bounding boxes, and classifications.  Requires some post processing of results to narrow down relevant boxes.  |

# Caffe Misc Networks for Neural Compute devices
|Network| Description |
|---------------------|-------------|
|TBD | TBD|
