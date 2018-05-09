# rapid-image-classifier

Perform image classification on a large number of images, using deep neural networks (DNN) on Intel® Movidius™ Neural Compute Stick (NCS). This sample code can be used to deploy various pre-trained DNNs such as GoogLeNet, SqueezeNet, MobileNet(s), Inception, etc. You can read more about this project, and a step-by-step guide of how to build such an app on <a href="https://movidius.github.io/blog/ncs-image-classifier/">NCS developer blog</a>.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running this example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/rapid-image-classifier/
make run
~~~
 
When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
==============================================================
Predicted 512_Amplifier.jpg as CD player in 111.30 ms with 52.0% confidence.
Predicted 512_Ball.jpg as soccer ball in 90.82 ms with 99.9% confidence.
Predicted 512_Cellphone.jpg as cellular telephone, cellular phone, cellphone, cell, mobile phone in 90.61 ms with 85.4% confidence.
Predicted 512_ElectricGuitar.jpg as electric guitar in 90.68 ms with 89.4% confidence.
Predicted 512_InkjetPrinter.jpg as printer in 91.58 ms with 86.0% confidence.
Predicted 512_LaserPrinter.jpg as photocopier in 90.67 ms with 95.9% confidence.
...
...
...
==============================================================
~~~

## Configuring this example

This example runs GoogLeNet by default, but you can configure it run other pre-trained deep neural networks. Below are some example commands:

AlexNet (Caffe)
~~~
python3 rapid-image-classifier.py --graph ../../caffe/AlexNet/graph --dim 227 227
~~~

SqueezeNet (Caffe)
~~~
python3 rapid-image-classifier.py --graph ../../caffe/SqueezeNet/graph --dim 227 227
~~~

Mobilenet (Tensorflow)
~~~
python3 rapid-image-classifier.py --graph ../../tensorflow/mobilenets/graph --labels ../../tensorflow/mobilenets/model/labels.txt --mean 127.5 --scale 0.00789 --dim 224 224 --colormode="RGB"
~~~

Inception (Tensorflow)
~~~
python3 rapid-image-classifier.py --graph ../../tensorflow/inception/model/v3/graph --labels ../../tensorflow/inception/model/v3/labels.txt --mean 127.5 --scale 0.00789 --dim 299 299 --colormode="RGB"
~~~

## Customizing this example

You can use this project as a template for your custom image classifier app. Below are some tips to help customize the example.

1. Before attemping to customize, check if the built-in options would suffice. Run `python3 rapid-image-classifier.py -h` to list all available options.
2. Steps 1, 2 and 5 are common across all Neural Compute Stick apps, so you can re-use those fuctions without modifications.
3. Step 3, 'Pre-process the images' is probably the most customizable function. As the name suggests, you can include all image pre-processing tasks in this function. Ex. if you don't want to warp the input image, just crop it before calling `skimage.transform.resize`.
4. Step 4 should be modified only if there is a need to change the way inference results are read and printed.
