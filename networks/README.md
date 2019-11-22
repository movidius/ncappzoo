# Neural Networks for the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2 (or original NCS) with OpenVINO<sup><sup><sup>™</sup></sup></sup> toolkit
This directory contains multiple subdirectories. Each subdirectory contains software, data, and instructions that pertain to using a specific neural network (based on any framework) with a Neural Compute device such as the Intel<sup><sup><sup>®</sup></sup></sup> NCS 2.  Along with the trained network itself, examples are provided via Makefile that show how the OpenVINO Model Optimizer can be used to compile the network to Intermediate Representation (IR) and also how to create a program that uses that IR model for inferencing.  The sections below are categorized by network type and include a brief explaination of each network.

This directory should be a preferred location for neural networks rather than the caffe or tensorflow directories.  The caffe and tensorflow directories are legacy directories so new networks should created in this directory.

# Image Classification Networks for Neural Compute devices
|Image Classification Network| Description |
|---------------------|-------------|
|[age_gender_net](age_gender_net/README.md) |Network that classifies a face image into age ranges. |
|[alexnet](alexnet/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[googlenet_v1](googlenet_v1/README.md) |BAIR/BLVC GoogleNet is a network based on [googlenet_v1](https://arxiv.org/abs/1409.4842), the winner of ILSVRC 2014, that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v1](inception_v1/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v2](inception_v2/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v3](inception_v3/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[inception_v4](inception_v4/README.md) |Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[resnet_50](resnet_50/README.md) |[Deep Residual network](https://arxiv.org/pdf/1512.03385.pdf)  with 50 layers that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[mnist](mnist/README.md) |Network that classifies handwritten digits.  This network is based on  tensorflow mnist_deep.  This project also includes an option to train the network yourself.|
|[mobilenets](mobilenets/README.md) |The mobilenets ([as described in the MobileNets Paper](https://arxiv.org/abs/1704.04861)) are small, low-latency, low-power Convolutional Neural Networks for Mobile Vision Applications.  They are parameterized for a variety of different uses.  Multiple trained networks with different parmameter values are compiled in this directory. |
|[squeezenet_v1.0](squeezenet_v1.0/README.md) |Accuracy similar to AlexNet with many fewer parameters and small model size as described int the [squeezenet paper](https://arxiv.org/abs/1602.07360). Network that classifies images based on the 1000 categories described in [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). |
|[ssd_inception_v2_traffic_light](ssd_inception_v2_traffic_light/README.md) |Single Shot Detector with inception v2 that was trained on 2 different labels on traffic light (green and red).  |


# Object Detection Networks for Neural Compute devices
|Object Detection Network| Description |
|---------------------|-------------|
|[face_detection_retail_0004](face_detection_retail_0004/README.md) |This is a nework that is trained to find faces in general.  [More information specific to this network is available.](https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-retail-0004/description/face-detection-retail-0004.md) |
|[ssd_inception_v2_gesture](ssd_inception_v2_gesture/README.md) |Single Shot Detector with inception v2 that was trained on 7 different 7 different hand gestures.  |
|[ssd_inception_v2_food](ssd_inception_v2_food/README.md) |Single Shot Detector with inception v2 that was trained on 10 different foods.  |
|[ssd_mobilenet_v1_caffe](ssd_mobilenet_v1_caffe/README.md)|MobileNet Single Shot Detector takes an image, detects the 20 PASCAL object classes as specified in the ([Visual Object Classes Challenges](http://host.robots.ox.ac.uk/pascal/VOC/)), their bounding boxes, and classifications. |
|[tiny_yolo_v1](tiny_yolo_v1/README.md) |This Tiny You Only Look Once model is based on [tiny-yolo v1 DarkNet model ](https://pjreddie.com/darknet/yolov1/).  Given an image, detects the 20 PASCAL object classes as specified in the ([Visual Object Classes Challenges](http://host.robots.ox.ac.uk/pascal/VOC/)), their bounding boxes, and classifications.  Requires some post processing of results to narrow down relevant boxes.  |
|[tiny_yolo_v2](tiny_yolo_v2/README.md) |This Tiny You Only Look Once model is based on [tiny-yolo v2 DarkNet model ](https://pjreddie.com/darknet/yolov2/).  Given an image, detects the 20 PASCAL object classes as specified in the ([Visual Object Classes Challenges](http://host.robots.ox.ac.uk/pascal/VOC/)), their bounding boxes, and classifications.  Requires some post processing of results to narrow down relevant boxes.  |
|[tiny_yolo_v3](tiny_yolo_v3/README.md) |Tiny Yolo (You Only Look Once) v3 network.  For more information see [https://github.com/mystic123/tensorflow-yolo-v3.git](https://github.com/mystic123/tensorflow-yolo-v3.git)  |



# Misc Networks for Neural Compute devices
|Network| Description |
|---------------------|-------------|
|[facenet](facenet/README.md) |FaceNet is a nework that is trained to find and quantify landmarks on faces in general.  By comparing the face landmark quantification values (network inference output) on two images, it is possible to determine how likely the two faces are of the same person.  This is based on [work by David Sandberg](https://github.com/davidsandberg/facenet).  |
|[segmantic segmentation adas 0001](semantic_segmentation_adas_0001/README.md) | Semantic segmentation adas 0001 is a nework that is trained to do semantic segmentation on 20 different classes. [More information specific to this networkis avaialble.](https://docs.openvinotoolkit.org/2019_R1/_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html)  |
