# live-image-classifier-PiCam

Perform image classification on a LIVE camera feed using deep neural networks (DNNs) on Intel® Movidius™ Neural Compute Stick (NCS). 
This project was used Raspberry Pi Camera Module on Raspberry Pi. 


## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>
3. Raspberry Pi Camera Module

## Running this example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/live-image-classifier-PiCam/
make run
~~~

When the application runs normally and is able to connect to the NCS device, you will see a live feed from your camera with a virtual box drawn over the feed. Place an item/object within this box to see inference results. You can hit 'Q' or 'q' at any time to quit the app.
 
## Configuring this example

This example runs GoogLeNet by default, but you can configure it run other pre-trained deep neural networks. Below are some example commands:

AlexNet (Caffe)
~~~
python3 live-image-classifier.py --graph ../../caffe/AlexNet/graph --dim 227 227
~~~

SqueezeNet (Caffe)
~~~
python3 live-image-classifier.py --graph ../../caffe/SqueezeNet/graph --dim 227 227
~~~

Mobilenet (Tensorflow)
~~~
python3 live-image-classifier.py --graph ../../tensorflow/mobilenets/model/graph --labels ../../tensorflow/mobilenets/model/labels.txt --mean 127.5 --scale 0.00789 --dim 224 224 --colormode="RGB"
~~~

Inception (Tensorflow)
~~~
python3 live-image-classifier.py --graph ../../tensorflow/inception/model/v3/graph --labels ../../tensorflow/inception/model/v3/label.txt --mean 127.5 --scale 0.00789 --dim 299 299 --colormode="RGB"
~~~

## Customizing this example

You can use this project as a template for your custom image classifier app. Below are some tips to help customize the example.

1. Before attemping to customize, check if the built-in options would suffice. Run `python3 live-image-classifier.py -h` to list all available options.
2. Steps 1, 2 and 5 are common across all Neural Compute Stick apps, so you can re-use those fuctions without modifications.
3. Step 3, 'Pre-process the images' is probably the most customizable function. As the name suggests, you can include all image pre-processing tasks in this function. Ex. if you don't want to warp the input image, just crop it before calling `skimage.transform.resize`.
4. Step 4 should be modified only if there is a need to change the way inference results are read and printed.
