# realtime-image-classifier: Real-time image classification using DNNs on Intel® Movidius™ Neural Compute Stick (NCS)

This directory contains a python3 script that shows how perform inference on a LIVE camera feed using deep neural networks on the Intel Movidius Neural Compute Stick

## Prerequisites

This code example requires that the following components are available:
1. Movidius Neural Compute Stick
2. Movidius Neural Compute SDK
3. Python3
4. A computer with web camera


## Running the Example
To run the example code do the following :
1. Open a terminal and change directory to the realtime-image-classifier base directory
2. Type the following command in the terminal: make run 

When the application runs normally and is able to connect to the NCS device, you will see a live feed from your camera with a virtual box drawn over the feed. Place an item/object within is box to see inference results.

