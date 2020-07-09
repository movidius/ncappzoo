---
layout: default
title: Apps
permalink: /apps/
nav_order: 2
has_children: true
has_toc: false
---

# Applications for the Intel&reg; NCS 2 (or original NCS) with OpenVINO&trade; toolkit

This directory contains subdirectories for applications that use the Intel&reg; NCS 2 via the OpenVINO&trade; toolkit.  Typically the applications here make use of one or more of the neural networks in the repository.  They are also intended to be more involved and provide more of a real world application for the networks rather than simply serving as an example of the technology.
The sections below are categorized by application type and present the currently available applications in the repository in succinct lists.

Each application directory has a README that explains how to build and run it, as well as a Makefile that automates the steps for you.  The links in the tables below will take you to the README files for each application.

## Image Classification Applications
Image classification applications typically use one of the image classification networks in the repository to classify an image as to it's likeliness to be in each of the classes on which a network was trained.
For a step by step tutorial on how to build an image classification network look at [Build an Image Classifier in 5 steps](https://movidius.github.io/blog/ncs-image-classifier/) at the Intel® Movidius™ Neural Compute Stick Blog

|Image Classification Application| Description |++++++Thumbnail++++++|
|---------------------|-------------|-------|
|[classifier_flash](classifier_flash.html) | Python<br>Multiple Networks<br>Classifies images from filesystem and displays them one by one in GUI| ![](../images/classifier_flash.jpg)|
|[classifier_grid](classifier_grid/html) | Python<br>Multiple Networks<br>Classifies images from filesystem and displays a large grid highlighting each image as its classified. This application shows how to get the best performance out of the NCS and NCS 2 devices. | ![](../images/classifier_grid.jpg)|
|[classifier-gui](classifier_gui.html) | Python<br>Multiple Network<br>GUI to select network and image to classify.|![](../images/classifier_gui.jpg)|
|[gender_age](gender_age.html) | C++<br>[Caffe AgeNet](../networks/age_gender_net.html), [GenderNet](../networks/age_gender_net.html), [face-detection-retail.0004](../networks/face_detection_retail_0004.html)<br>Uses the face detection and Age-Gender Network to predict age and gender of people in a live camera feed. The camera feed is displayed with a box overlayed around the faces and a label for the age and gender of the person. |![](../images/gender_age.jpg)|
|[mnist_calc](mnist_calc.html) |Python<br>TensorFlow [mnist_deep](../networks/mnist.html) Network<br>Handwriting calculator based on mnist.  Does digit detection on writen numbers to determine the equation and calculates the results.  A fun project to do with a Raspberry Pi and a touch screen!  |![](../images/mnist_calc.jpg)|
|[simple_classifier_cpp](simple_classifier_cpp.html) | C++<br>Multiple Networks<br>Application reads a single image from the filesystem and does an image classification inference on that image. Takes the image, the network and a labels file on the commandline|![](../images/simple_class_cpp.jpg)|
|[simple_classifier_py](simple_classifier_py.html) | Python<br>Multiple Networks<br>Application reads a single image from the filesystem and does an image classification inference on that image. Takes the image, the network and a labels file on the commandline|![](../images/simple_class_python.jpg)|
|[simple_classifier_py_camera](simple_classifier_py_camera.html) | Python<br>Multiple Networks<br>Application reads a video stream from a camera and does image classification inference on the stream continually updating the top result.|![](../images/simple_class_py_camera.jpg)|


## Object Detection Applications
Object detection applications make use of one of the [object detection networks](TODO) in the repository to detect objects within an image.  The object detection networks typically determine where objects are within the image as well as what type of objects they are.

|Object Detection Application| Description |+++++Thumbnail+++++ |
|---------------------|-------------|-------|
|[birds](birds.html) | Python<br>[Caffe Tiny Yolo](../networks/tiny_yolo_v1.html), [GoogLeNet](../networks/googlenet_v1.html<br>Detects and identifies birds in photos by using Yolo Tiny to identify birds in general and then GoogLeNet to further classify them. Displays images with overlayed rectangles bird classification. |![](../images/birds.jpg)|
|[chef_ai](chef_ai.html) |Python / Flask<br>[SSD Inception v2 (food)](../networks/ssd_inception_v2_food.html)<br> Detects food items within an image or from the camera and suggests recipes for those items. |[](../images/chef_ai.gif)   |
|[driving_pi](driving_pi.html) | Python<br>Multiple Networks<br>Autonomous LEGO car driving by detecting road and traffic light through a camera. car will stop once see the red light and moves forward on green. It also slowing speed on the yellow light. In this demo, it uses both networks, object detection for traffic lights and road classification for detecting road.|![](../images/driving_pi.jpg)|
|[realsense_object_distance_detection](realsense_object_distance_detection.html) | C++<br>[Caffe SSD Mobilenet](../networks/ssd_mobilenet_v1_caffe.html)(../<br>Detects different classes of objects (including people and cars) and uses the Intel Realsense camera to detect the distance to that object. |![](../images/realsense_object_distance_detection.jpg)|
|[realsense_segmentation](realsense_segmentation.html) | C++<br>[Semantic Segmentation adas 0001](../networks/semantic_segmentation_adas_0001.md)<br>Colorize 20 different classes of objects (including people and cars) and uses the Intel Realsense camera to detect the distance to that object. | ![](../images/realsense_segmentation.png)|

## Misc Applications
Miscellaneous applications use the OpenVINO toolkit in various ways that don't fit into any of the above categories but can still be interesting.

|Misc Application| Description |+++++Thumbnail+++++ |
|---------------------|-------------|-------|
|[benchmark_ncs](benchmark_ncs.html) | Python<br>Multiple Networks<br>Outputs FPS numbers for networks in the repository that take images as input. The number of NCS devices to use for the FPS numbers can be specified on the commandline.|![](../images/benchmark_ncs.jpg)|
|[ncs_digital_sign](ncs_digital_sign/README.md) | C++<br>Multiple Networks<br>Application for a digital sign that displays advertisements targeted towards the demographic (gender and age) of the person viewing the sign|![](../images/ncs_digital_sign.gif)|
|[gesture_piarm](gesture_piarm.html) | Python<br>[SSD Inception v2 (gesture)](../networks/ssd_inception_v2_gesture.html)<br>Controls a robotic arm using different hand gestures - made for Raspberry Pi!|![](../images/Arm_Motion_Demo.gif)

