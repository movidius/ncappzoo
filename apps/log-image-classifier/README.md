# log-image-classifier

Logs results of an image classifier into a comma-separated values (CSV) file. The classifier uses deep neural networks (DNNs) on Intel® Movidius™ Neural Compute Stick (NCS) to run inferences sequentially (and recursively) on all images within a folder.

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>

## Running the Example

~~~
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/movidius/ncappzoo
cd ~/workspace/ncappzoo/apps/log-image-classifier/
make run
~~~

When the application runs normally and is able to connect to the NCS device, the output will be similar to this:

~~~
Pre-processing images...

Performing inference on a lot of images...

Inference complete! View results in ./inferences.csv.
~~~



