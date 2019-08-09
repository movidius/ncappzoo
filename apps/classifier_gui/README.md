# classifier GUI: Image classification using DNNs on the Intel Neural Compute Stick 2 (NCS/NCS2)

This directory contains a python3 example that shows how to classify images using deep neural networks on the Intel Neural Compute Stick 2 using a simple GUI. This sample can be used with [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet), [SqueezeNet V1.0](https://github.com/DeepScale/SqueezeNet), and [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet). All models are sourced from the [Open Model Zoo](https://github.com/opencv/open_model_zoo).

The provided Makefile does the following:
1. Builds the IR files using the model files from [Open Model Zoo](https://github.com/opencv/open_model_zoo).
2. Copies the IR files from the model directory to the project base directory.
3. Runs the sample.

## Running the Example
To run the example code do the following :
1. Open a terminal and change directory to the project base directory
2. Type the following command in the terminal: ```make run``` 

When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
classifier_gui: Running the sample...
python3 classifier_gui.py

Starting application...
   - Plugin:       Myriad
   - IR File:      ../../caffe/GoogLeNet/googlenet-v1.xml
   - Input Shape:  [1, 3, 224, 224]
   - Output Shape: [1, 1000]
   - Labels File:  ../../data/ilsvrc12/synset_labels.txt
   - Mean File:    None
   - Image File:    ../../data/images/nps_mug.png

 **********  Results  ***********

 Prediction is 91.0% coffee mug
~~~

You should also see the image on which inference was performed.


## Prerequisites
This program requires:
- 1 NCS device
- OpenVINO 2019 R2 Toolkit
- OpenCV 3.3 with Video for Linux (V4L) support and associated Python bindings*.
- Scikit-image
- PIL (Python Imaging library) with Image and Imagetk modules

**Note:** To install Scikit-image, use the following command: ```sudo apt-get install -y python3-pil.imagetk```

**Note:** To install PIL with ImageTk, use the following command: ```pip3 install scikit-image --user```

*It may run with older versions but you may see some glitches such as the GUI Window not closing when you click the X in the title bar, and other key binding issues.

Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make run
Runs the sample application.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application.

### make data
Gathers all of the required data need to run the sample.

### make deps
Builds all of the dependencies needed to run the sample.

### make install_reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 
 
### make clean
Removes all the temporary files that are created by the Makefile.

