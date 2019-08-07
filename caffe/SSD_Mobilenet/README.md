# ssd mobilenet
## Introduction
The [ssd mobilenet](https://github.com/chuanqi305/MobileNet-SSD) network can be used for object detection and can detect 20 different types of objects. This model was pretrained and more information can be found at: https://github.com/chuanqi305/MobileNet-SSD. The list of objects that it can detect are:

```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

The provided Makefile does the following

1. Downloads the prototxt and caffe weight files using the model downloader from the [Open Model Zoo](https://github.com/opencv/open_model_zoo)
2. Takes an image and runs an inference using the ssd mobilenet model.


## Running this Example
~~~
make run
~~~

## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make run
Runs a sample application with the network.

### make run_py
Runs the face_detect.py python script which sends a single image to the Neural Compute Stick and receives and displays the inference results.

### make help
Shows makefile possible targets and brief descriptions. 

### make all
Makes the follow items: deps, data.

### make compile_model
Uses the network description and the trained weights files to generate an IR (intermediate representation) format file.  This file is later loaded on the Neural Compute Stick where the inferences on the network can be executed.  

### make clean
Removes all the temporary and target files that are created by the Makefile.

