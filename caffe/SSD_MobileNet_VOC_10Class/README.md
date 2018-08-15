# Introduction
This directory includes the SSD MobileNet network and example code.  It is a caffe based object recognition and classification network.  This project was created to show how to train a SSD mobilenet using Caffe, then deploy it on Movidius NCS. There are 10 classes including background defined in `lable.txt`.  Please take a look at this github repository for more information: https://github.com/yuanyuanli85/caffe/tree/ssd_x10.

The provided Makefile does the following:
1. Downloads network prototxt and caffemodel.
2. Compiles the network files into a graph file for the NCS device.
3. Runs the provided run.py program which creates a GUI window that shows the results of the network for a single image.
4. Downloads required datasets, generates lmdb and train network.

# Prerequisites
This program requires:
- 1 NCS device
- NCSDK 1.11 or greater
	- run 'make install' and then 'make examples'
- OpenCV
- Caffe if you want to train this model yourself.

# Install Caffe
This setp is required ONLY if you want to train the network by yourself. You can skip this setp if you want to use pre-trained model. This caffe is a customized version to support this case.

```
git clone https://github.com/yuanyuanli85/caffe.git
cd caffe
git checkout ssd_x10
cp Makefile.config.example Makefile.config
make all -j8
make pycaffe
export CAFFE_PATH=/PATH/TO/YOUR/CAFFE
```

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets.

## make all
Builds and/or gathers all the required files needed to run the application except the ncsdk.  This must be done as a separate step.

## make run_py
Runs the provided python program which runs a single inference on a single image and displays the output in a GUI window as well as on the console.

## make clean
Removes all the temporary files that are created by the Makefile

## make data
Downloads the required dataset, and generate the lmdb for training. If everything goes well, the two lmdbs `./data/VOCdevkit/lmdb/trainval_lmdb` and `./data/VOCdevkit/lmdb/test_lmdb` will be generated.

## make train
Train the network, and caffe snapshots will be stored in `./mobilenet-ssd-1.0-x10/snapshot`.
