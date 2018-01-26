# rapid-image-classifier

Perform image classification on a large number of images, using DNNs on Intel® Movidius™ Neural Compute Stick (NCS). This sample code was used to validate a Dogs vs Cats classifier built using a customized version of GoogLeNet. You can read more about this project (and a step-by-step guide) on <a href="https://movidius.github.io/blog/deploying-custom-caffe-models/">NCS developer blog</a>. 

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running the Example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/rapid-image-classifier/
make run
~~~

When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
Pre-processing images...
Prediction for main-street-in-chinatown.jpg: streetcar, tram, tramcar, trolley, trolley car with 33.4% confidence in 48.90 ms
Prediction for pic_011.jpg: cheetah, chetah, Acinonyx jubatus with 56.4% confidence in 39.19 ms
Prediction for pic_071.jpg: African elephant, Loxodonta africana with 94.8% confidence in 39.50 ms
Prediction for pic_035.jpg: brain coral with 44.3% confidence in 39.15 ms
Prediction for pic_036.jpg: jellyfish with 99.7% confidence in 39.47 ms
Prediction for pic_017.jpg: English setter with 46.5% confidence in 39.41 ms
Prediction for pic_013.jpg: beaver with 34.7% confidence in 39.40 ms
...
...
...
~~~



