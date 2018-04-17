# image-classifier

Perform image classification using deep neural networks (DNNs) on Intel® Movidius™ Neural Compute Stick (NCS). The <a href="https://movidius.github.io/blog/ncs-image-classifier/">NCS developer blog</a> has a step by step tutorial on how to build this project, and also has a detailed explanation of the source code.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running this example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/image-classifier/
make run
~~~
 
When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
==============================================================
Top predictions for cat.jpg
Execution time: 95.0071ms
--------------------------------------------------------------
40.4%	n02123159 tiger cat
32.7%	n02123045 tabby, tabby cat
8.9%	n02124075 Egyptian cat
5.0%	n02127052 lynx, catamount
1.2%	n04074963 remote control, remote
1.1%	n02971356 carton
==============================================================
~~~

You should also see the image on which inference was performed.

## Configuring this example

This example runs GoogLeNet by default, but you can configure it run other pre-trained deep neural networks. Below are some example commands:

AlexNet (Caffe)
~~~
python3 image-classifier.py --graph ../../caffe/AlexNet/graph --dim 227 227 --image ../../data/images/pic_053.jpg
~~~

SqueezeNet (Caffe)
~~~
python3 image-classifier.py --graph ../../caffe/SqueezeNet/graph --dim 227 227 --image ../../data/images/pic_053.jpg
~~~

Mobilenet (Tensorflow)
~~~
python3 image-classifier.py --graph ../../tensorflow/mobilenets/model/graph --labels ../../tensorflow/mobilenets/model/labels.txt --mean 127.5 --scale 0.00789 --dim 224 224 --colormode="RGB" --image ../../data/images/pic_053.jpg 
~~~

Inception (Tensorflow)
~~~
python3 image-classifier.py --graph ../../tensorflow/inception/model/v3/graph --labels ../../tensorflow/inception/model/v3/labels.txt --mean 127.5 --scale 0.00789 --dim 299 299 --colormode="RGB" --image ../../data/images/pic_053.jpg 
~~~

## Customizing this example

You can use this project as a template for your custom image classifier app. Below are some tips to help customize the example.

1. Before attemping to customize, check if the built-in options would suffice. Run `python3 image-classifier.py -h` to list all available options.
2. Steps 1, 2 and 5 are common across all Neural Compute Stick apps, so you can re-use those fuctions without modifications.
3. Step 3, 'Pre-process the images' is probably the most customizable function. As the name suggests, you can include all image pre-processing tasks in this function. Ex. if you don't want to warp the input image, just crop it before calling `skimage.transform.resize`.
4. Step 4 should be modified only if there is a need to change the way inference results are read and printed.
